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
    Perform one meta-training step on source data using prototype-based approach
    
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
    
    # Ensure vocab has template_to_idx method - silently add if needed
    if not hasattr(vocab, 'template_to_idx'):
        vocab.template_to_idx = lambda template: vocab.word2id(str(template))
    
    # Process in smaller chunks to save memory
    max_batch_size = min(batch_size, 32)  # Reduced batch size
    
    # Divide data into support and query
    support_data = source_support_set[:max_batch_size]
    query_data = source_query_set[:max_batch_size]
    
    # Prepare support data
    support_tinst, support_labels = prepare_batch_for_training(support_data, vocab, verbose=False)
    
    # Create model inputs
    support_words = support_tinst.to(device)
    support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
    support_word_len = torch.sum(support_masks, dim=1).to(device)
    support_model_inputs = (support_words, support_masks, support_word_len)
    
    # Forward pass on support set - calculate prototype representation
    with torch.set_grad_enabled(True):
        # Get embeddings
        support_logits, _, support_embeddings = encoder(support_model_inputs)
        
        # Create prototype as mean of embeddings
        prototype = support_embeddings.mean(dim=0, keepdim=True)
        
        # Free memory
        del support_model_inputs, support_words, support_masks, support_word_len
        torch.cuda.empty_cache()
    
    # Calculate loss based on prototype and support logits
    support_loss = torch.nn.CrossEntropyLoss()(support_logits, support_labels.to(device))
    
    # Update model parameters
    optimizer.zero_grad()
    support_loss.backward()
    optimizer.step()
    
    # Clear memory
    del support_logits, support_embeddings
    torch.cuda.empty_cache()
    
    # Prepare query data
    query_tinst, query_labels = prepare_batch_for_training(query_data, vocab, verbose=False)
    
    # Create model inputs for query
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    query_model_inputs = (query_words, query_masks, query_word_len)
    
    # Forward pass on query set with no gradient
    with torch.no_grad():
        # Get embeddings
        query_logits, _, query_embeddings = encoder(query_model_inputs)
        
        # Calculate loss
        query_loss = torch.nn.CrossEntropyLoss()(query_logits, query_labels.to(device))
        
        # Free memory
        del query_model_inputs, query_words, query_masks, query_word_len, query_embeddings, query_logits, prototype
        torch.cuda.empty_cache()
    
    return query_loss.item()


def meta_test_step(target_support_set, target_query_set, encoder, optimizer, device, batch_size=32):
    """
    Perform one meta-testing step on target data using prototype-based approach
    
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
    
    # Ensure vocab has template_to_idx method - silently add if needed
    if not hasattr(vocab, 'template_to_idx'):
        vocab.template_to_idx = lambda template: vocab.word2id(str(template))
    
    # Process in smaller chunks to save memory
    max_batch_size = min(batch_size, 32)  # Reduced batch size
    
    # Use only normal samples from support set (like in train_mtalog)
    normal_support_set = [inst for inst in target_support_set[:max_batch_size] if 
                           hasattr(inst, 'label') and (inst.label == 0 or inst.label == "Normal")]
    
    # If no normal samples, use all samples
    if not normal_support_set:
        normal_support_set = target_support_set[:max_batch_size]
    
    # Prepare support data
    support_tinst, support_labels = prepare_batch_for_training(normal_support_set, vocab, verbose=False)
    
    # Create model inputs
    support_words = support_tinst.to(device)
    support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
    support_word_len = torch.sum(support_masks, dim=1).to(device)
    support_model_inputs = (support_words, support_masks, support_word_len)
    
    # Forward pass on support set to calculate prototype
    with torch.no_grad():
        # Get embeddings
        _, _, support_embeddings = encoder(support_model_inputs)
        
        # Create prototype as mean of embeddings
        prototype = support_embeddings.mean(dim=0, keepdim=True)
        
        # Free memory
        del support_model_inputs, support_words, support_masks, support_word_len, support_embeddings
        torch.cuda.empty_cache()
    
    # Prepare query data
    query_batch = target_query_set[:max_batch_size]
    query_tinst, query_labels = prepare_batch_for_training(query_batch, vocab, verbose=False)
    
    # Create model inputs for query
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    query_model_inputs = (query_words, query_masks, query_word_len)
    
    # Forward pass on query set with no gradient
    with torch.no_grad():
        # Get embeddings and logits
        query_logits, _, query_embeddings = encoder(query_model_inputs)
        
        # Calculate cosine similarity to prototype
        similarity = torch.nn.functional.cosine_similarity(query_embeddings, prototype, dim=1)
        
        # Calculate loss
        query_loss = torch.nn.CrossEntropyLoss()(query_logits, query_labels.to(device))
        
        # Make predictions
        y_pred = torch.argmax(query_logits, dim=1).cpu().numpy()
        y_true = query_labels.cpu().numpy()
        
        # Free memory
        del query_model_inputs, query_words, query_masks, query_word_len, query_embeddings, query_logits, prototype, similarity
        torch.cuda.empty_cache()
    
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
    Evaluate the trained model on test data using a memory-efficient approach
    
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
    
    # Process in chunks to reduce memory usage
    all_predictions = []
    all_true_labels = []
    
    chunk_size = min(batch_size, 32)  # Smaller batch size to prevent OOM
    num_chunks = (len(test_data) + chunk_size - 1) // chunk_size
    
    logger.info(f"Evaluating on {len(test_data)} instances in {num_chunks} chunks")
    
    total_loss = 0.0
    
    for chunk_idx in range(num_chunks):
        # Get chunk of data
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(test_data))
        chunk_data = test_data[start_idx:end_idx]
        
        # Skip empty chunks
        if not chunk_data:
            continue
            
        logger.info(f"Processing evaluation chunk {chunk_idx+1}/{num_chunks}, size: {len(chunk_data)}")
        
        # Prepare test data for this chunk
        test_tinst, test_labels = prepare_batch_for_training(chunk_data, vocab)
        
        # Create model inputs
        test_words = test_tinst.to(device)
        test_masks = torch.ones_like(test_words, dtype=torch.float, device=device)
        test_word_len = torch.sum(test_masks, dim=1).to(device)
        test_model_inputs = (test_words, test_masks, test_word_len)
        
        # Forward pass with no gradient
        with torch.no_grad():
            # Get embeddings and logits
            test_logits, _, test_embeddings = encoder(test_model_inputs)
            
            # Calculate loss
            chunk_loss = torch.nn.CrossEntropyLoss()(test_logits, test_labels.to(device))
            total_loss += chunk_loss.item()
            
            # Make predictions
            chunk_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            chunk_true = test_labels.cpu().numpy()
            
            # Add to full results
            all_predictions.extend(chunk_preds)
            all_true_labels.extend(chunk_true)
            
            # Free memory
            del test_words, test_masks, test_word_len, test_model_inputs
            del test_logits, test_embeddings, chunk_loss
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_true_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Calculate average loss
    avg_loss = total_loss / num_chunks if num_chunks > 0 else 0
    
    # Log results
    logger.info(f"Test Loss: {avg_loss:.4f}")
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
    Make predictions for new log sequences using a memory-efficient approach
    
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
    
    # Process in chunks to reduce memory usage
    all_predictions = []
    all_scores = []
    
    chunk_size = 32  # Smaller batch size to prevent OOM
    num_chunks = (len(log_sequences) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        # Get chunk of data
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(log_sequences))
        chunk_data = log_sequences[start_idx:end_idx]
        
        # Skip empty chunks
        if not chunk_data:
            continue
        
        # Prepare inputs for this chunk
        chunk_tinst, _ = prepare_batch_for_training(chunk_data, vocab)
        
        # Prepare inputs in correct format
        chunk_words = chunk_tinst.to(device)
        chunk_masks = torch.ones_like(chunk_words, dtype=torch.float, device=device)
        chunk_word_len = torch.sum(chunk_masks, dim=1).to(device)
        
        # Process through model in smaller chunks
        with torch.no_grad():
            # Create model inputs
            model_inputs = (chunk_words, chunk_masks, chunk_word_len)
            
            # Get predictions
            logits, _, embeddings = encoder(model_inputs)
            
            # Get raw scores (probability of being anomalous)
            probs = torch.softmax(logits, dim=1)
            anomaly_scores = probs[:, 1].cpu().numpy()  # Probability of class 1 (anomaly)
            
            # Make binary predictions
            chunk_preds = (anomaly_scores > threshold).astype(int)
            
            # Add to full results
            all_predictions.extend(chunk_preds)
            all_scores.extend(anomaly_scores)
            
            # Free memory
            del chunk_words, chunk_masks, chunk_word_len, model_inputs
            del logits, embeddings, probs
            torch.cuda.empty_cache()
    
    return np.array(all_predictions), np.array(all_scores) 