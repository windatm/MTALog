#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and evaluation functions for MTALog
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils.data_processing import (
    encode_sequences, 
    prepare_batch_for_training,
    calculate_metrics,
    create_embedding_lookup
)


def ensure_vocab_has_template_to_idx(vocab):
    """Add a template_to_idx method to the vocab object if it doesn't have one."""
    if not hasattr(vocab, 'template_to_idx'):
        # Add the method dynamically
        vocab.template_to_idx = lambda template: vocab.word2id(str(template))
    return vocab


def meta_train_step(source_support_set, source_query_set, encoder, optimizer, device, batch_size=32):
    """
    Perform one meta-training step on source data
    
    Args:
        source_support_set: Support set from source domain
        source_query_set: Query set from source domain
        encoder: Neural network encoder model
        optimizer: Optimizer for parameter updates
        device: Device to run computations on
        batch_size: Batch size for training
        
    Returns:
        loss: Training loss value
    """
    # Set model to training mode
    encoder.train()
    
    # Get vocab with fallback
    if hasattr(encoder, 'vocab'):
        vocab = encoder.vocab
    else:
        # Try to extract vocab from first instance in support set
        vocab = getattr(source_support_set[0], 'vocab', None)
        if vocab is None:
            raise AttributeError("No vocab found in encoder or support set. Cannot proceed with training.")
    
    # Ensure vocab has template_to_idx method
    vocab = ensure_vocab_has_template_to_idx(vocab)
    
    # Prepare support and query data
    support_tinst, support_labels = prepare_batch_for_training(source_support_set, vocab)
    query_tinst, query_labels = prepare_batch_for_training(source_query_set, vocab)
    
    # Prepare inputs in the correct format
    # The model expects a tuple of (words, masks, word_len)
    support_words = support_tinst.to(device)
    support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
    support_word_len = torch.sum(support_masks, dim=1).to(device)
    
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    
    # Forward pass on support set
    support_model_inputs = (support_words, support_masks, support_word_len)
    support_logits, _, support_embeddings = encoder(support_model_inputs)
    
    # Using standard cross entropy loss for classification
    criterion = torch.nn.CrossEntropyLoss()
    support_loss = criterion(support_logits, support_labels.to(device))
    
    # Update model parameters with support set (inner loop update)
    optimizer.zero_grad()
    support_loss.backward()
    optimizer.step()
    
    # Forward pass on query set with updated parameters
    query_model_inputs = (query_words, query_masks, query_word_len)
    query_logits, _, query_embeddings = encoder(query_model_inputs)
    
    # Calculate loss on query set
    query_loss = criterion(query_logits, query_labels.to(device))
    
    return query_loss.item()


def meta_test_step(target_support_set, target_query_set, encoder, optimizer, device, batch_size=32):
    """
    Perform one meta-testing step on target data
    
    Args:
        target_support_set: Support set from target domain
        target_query_set: Query set from target domain
        encoder: Neural network encoder model
        optimizer: Optimizer for parameter updates
        device: Device to run computations on
        batch_size: Batch size for training
        
    Returns:
        loss: Test loss value
        metrics: Performance metrics dictionary
    """
    # Set model to evaluation mode
    encoder.eval()
    
    # Get vocab with fallback
    if hasattr(encoder, 'vocab'):
        vocab = encoder.vocab
    else:
        # Try to extract vocab from first instance in support set
        vocab = getattr(target_support_set[0], 'vocab', None)
        if vocab is None:
            raise AttributeError("No vocab found in encoder or support set. Cannot proceed with testing.")
    
    # Ensure vocab has template_to_idx method
    vocab = ensure_vocab_has_template_to_idx(vocab)
    
    # Prepare support and query data
    support_tinst, support_labels = prepare_batch_for_training(target_support_set, vocab)
    query_tinst, query_labels = prepare_batch_for_training(target_query_set, vocab)
    
    # Prepare inputs in the correct format
    # The model expects a tuple of (words, masks, word_len)
    support_words = support_tinst.to(device)
    support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
    support_word_len = torch.sum(support_masks, dim=1).to(device)
    
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    
    # Forward pass on support set (for adaptation)
    with torch.no_grad():
        support_model_inputs = (support_words, support_masks, support_word_len)
        support_logits, _, support_embeddings = encoder(support_model_inputs)
    
    # Since the optimizer doesn't have an adapt method, we'll use a simple approach
    # for adaptation without modifying the parameters
    
    # Forward pass on query set
    with torch.no_grad():
        query_model_inputs = (query_words, query_masks, query_word_len)
        query_logits, _, query_embeddings = encoder(query_model_inputs)
        
        # Calculate loss using cross entropy
        criterion = torch.nn.CrossEntropyLoss()
        query_loss = criterion(query_logits, query_labels.to(device))
        
        # Make predictions
        y_pred = torch.argmax(query_logits, dim=1).cpu().numpy()
        y_true = query_labels.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    return query_loss.item(), metrics


def train_model(
    source_systems, 
    source_support_sets, 
    source_query_sets,
    target_support_set,
    target_query_set,
    source_encoders,
    target_encoder,
    optimizer,
    device,
    num_epochs,
    batch_size,
    output_model_dir,
    logger
):
    """
    Train the model using meta-learning approach
    
    Args:
        source_systems: List of source system names
        source_support_sets: Dictionary of support sets for each source system
        source_query_sets: Dictionary of query sets for each source system
        target_support_set: Support set for target system
        target_query_set: Query set for target system
        source_encoders: Dictionary of encoders for each source system
        target_encoder: Encoder for target system
        optimizer: Optimizer for parameter updates
        device: Device to run computations on
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        output_model_dir: Directory to save model
        logger: Logger instance
        
    Returns:
        best_model: Best model based on validation performance
        best_f1: Best F1 score achieved during training
    """
    # Training statistics
    start_time = time.time()
    best_f1 = 0
    best_model = None
    
    # Defensive check for vocab in encoders
    # For target encoder
    if not hasattr(target_encoder, 'vocab') and hasattr(source_encoders[source_systems[0]], 'vocab'):
        logger.warning("Target encoder missing vocab attribute. Using vocab from first source encoder.")
        target_encoder.vocab = source_encoders[source_systems[0]].vocab
    
    # For source encoders
    for source_system in source_systems:
        if not hasattr(source_encoders[source_system], 'vocab'):
            logger.warning(f"Source encoder for {source_system} missing vocab attribute. Using target encoder's vocab.")
            # Try to get vocab from target encoder
            if hasattr(target_encoder, 'vocab'):
                source_encoders[source_system].vocab = target_encoder.vocab
            else:
                logger.error(f"No vocab available for {source_system} encoder. Training may fail.")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
        epoch_start_time = time.time()
        
        # Meta-training on source systems
        for source_system in source_systems:
            logger.info(f"Meta-training on {source_system}")
            support_set = source_support_sets[source_system]
            query_set = source_query_sets[source_system]
            encoder = source_encoders[source_system]
            
            # Train on batches
            total_loss = 0
            num_batches = len(support_set) // batch_size
            
            for i in range(num_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                
                support_batch = support_set[start_idx:end_idx]
                query_batch = query_set[start_idx:end_idx]
                
                # Perform meta-training step
                loss = meta_train_step(
                    support_batch, 
                    query_batch, 
                    encoder, 
                    optimizer, 
                    device, 
                    batch_size
                )
                total_loss += loss
            
            avg_loss = total_loss / num_batches
            logger.info(f"  {source_system} Average Loss: {avg_loss:.4f}")
        
        # Meta-testing on target system
        logger.info(f"Meta-testing on target system")
        loss, metrics = meta_test_step(
            target_support_set,
            target_query_set,
            target_encoder,
            optimizer,
            device,
            batch_size
        )
        
        # Log performance
        logger.info(f"  Target Loss: {loss:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = target_encoder.state_dict()
            
            # Save model checkpoint
            model_path = os.path.join(output_model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(best_model, model_path)
            logger.info(f"  New best model saved to {model_path}")
        
        # Log epoch time
        epoch_time = time.time() - epoch_start_time
        logger.info(f"  Epoch completed in {epoch_time:.2f} seconds")
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best F1 Score: {best_f1:.4f}")
    
    return best_model, best_f1


def evaluate_model(
    test_data,
    encoder,
    optimizer,
    device,
    batch_size,
    logger
):
    """
    Evaluate the trained model on test data
    
    Args:
        test_data: Test dataset
        encoder: Trained encoder model
        optimizer: Optimizer with loss function
        device: Device to run computations on
        batch_size: Batch size for evaluation
        logger: Logger instance
        
    Returns:
        metrics: Performance metrics dictionary
    """
    logger.info("=== Model Evaluation ===")
    
    # Set model to evaluation mode
    encoder.eval()
    
    # Get vocab with fallback
    if hasattr(encoder, 'vocab'):
        vocab = encoder.vocab
    else:
        # Try to extract vocab from first instance in test data
        vocab = getattr(test_data[0], 'vocab', None)
        if vocab is None:
            logger.error("No vocab found in encoder or test data. Evaluation may fail.")
            # Try to create a simple vocab as last resort
            from utils.vocab import Vocab
            vocab = Vocab()
    
    # Ensure vocab has template_to_idx method
    vocab = ensure_vocab_has_template_to_idx(vocab)
    
    # Prepare test data
    test_tinst, test_labels = prepare_batch_for_training(test_data, vocab)
    
    # Prepare inputs in the correct format
    test_words = test_tinst.to(device)
    test_masks = torch.ones_like(test_words, dtype=torch.float, device=device)
    test_word_len = torch.sum(test_masks, dim=1).to(device)
    
    # Forward pass
    with torch.no_grad():
        test_model_inputs = (test_words, test_masks, test_word_len)
        test_logits, _, test_embeddings = encoder(test_model_inputs)
        
        # Calculate loss using cross entropy
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = criterion(test_logits, test_labels.to(device))
        
        # Make predictions
        y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
        y_true = test_labels.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Log results
    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"True Positives: {metrics['tp']}")
    logger.info(f"False Positives: {metrics['fp']}")
    logger.info(f"False Negatives: {metrics['fn']}")
    logger.info(f"True Negatives: {metrics['tn']}")
    
    return metrics


def predict(
    log_sequences,
    encoder,
    template_lookup,
    device,
    threshold=0.5
):
    """
    Make predictions for new log sequences
    
    Args:
        log_sequences: List of log sequences to classify
        encoder: Trained encoder model
        template_lookup: Lookup table for template embeddings
        device: Device to run computations on
        threshold: Classification threshold
        
    Returns:
        predictions: Binary predictions (0 for normal, 1 for anomaly)
        scores: Anomaly scores for each sequence
    """
    # Set model to evaluation mode
    encoder.eval()
    
    # Get vocab with fallback
    if hasattr(encoder, 'vocab'):
        vocab = encoder.vocab
    else:
        # Try to extract vocab from first instance in log sequences
        vocab = getattr(log_sequences[0], 'vocab', None)
        if vocab is None:
            raise AttributeError("No vocab found in encoder or log sequences. Cannot make predictions.")
    
    # Ensure vocab has template_to_idx method
    vocab = ensure_vocab_has_template_to_idx(vocab)
    
    # Prepare inputs
    tinst, _ = prepare_batch_for_training(log_sequences, vocab)
    
    # Prepare inputs in correct format
    words = tinst.to(device)
    masks = torch.ones_like(words, dtype=torch.float, device=device)
    word_len = torch.sum(masks, dim=1).to(device)
    
    # Get predictions
    with torch.no_grad():
        model_inputs = (words, masks, word_len)
        logits, _, embeddings = encoder(model_inputs)
        
        # Get raw scores (probability of being anomalous)
        probs = torch.softmax(logits, dim=1)
        anomaly_scores = probs[:, 1].cpu().numpy()  # Probability of class 1 (anomaly)
        
        # Make binary predictions
        predictions = (anomaly_scores > threshold).astype(int)
    
    return predictions, anomaly_scores 