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
    
    # Prepare support and query data
    support_inputs, support_labels = prepare_batch_for_training(source_support_set, vocab)
    query_inputs, query_labels = prepare_batch_for_training(source_query_set, vocab)
    
    # Move data to device
    support_inputs = support_inputs.to(device)
    support_labels = support_labels.to(device)
    query_inputs = query_inputs.to(device)
    query_labels = query_labels.to(device)
    
    # Forward pass on support set
    support_embeddings = encoder(support_inputs)
    
    # Update model parameters with support set (inner loop update)
    optimizer.zero_grad()
    support_loss = optimizer.compute_loss(support_embeddings, support_labels)
    support_loss.backward()
    optimizer.step()
    
    # Forward pass on query set with updated parameters
    query_embeddings = encoder(query_inputs)
    query_loss = optimizer.compute_loss(query_embeddings, query_labels)
    
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
    
    # Prepare support and query data
    support_inputs, support_labels = prepare_batch_for_training(target_support_set, vocab)
    query_inputs, query_labels = prepare_batch_for_training(target_query_set, vocab)
    
    # Move data to device
    support_inputs = support_inputs.to(device)
    support_labels = support_labels.to(device)
    query_inputs = query_inputs.to(device)
    query_labels = query_labels.to(device)
    
    # Forward pass on support set (for adaptation)
    with torch.no_grad():
        support_embeddings = encoder(support_inputs)
    
    # Compute adapted parameters based on support set
    adapted_params = optimizer.adapt(support_embeddings, support_labels)
    
    # Forward pass on query set with adapted parameters
    with torch.no_grad():
        query_embeddings = encoder(query_inputs, params=adapted_params)
        query_loss = optimizer.compute_loss(query_embeddings, query_labels, params=adapted_params)
        
        # Make predictions
        y_pred = optimizer.predict(query_embeddings, adapted_params)
        y_true = query_labels.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred.cpu().numpy())
    
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
    
    # Prepare test data
    test_inputs, test_labels = prepare_batch_for_training(test_data, vocab)
    
    # Move data to device
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)
    
    # Forward pass
    with torch.no_grad():
        test_embeddings = encoder(test_inputs)
        test_loss = optimizer.compute_loss(test_embeddings, test_labels)
        
        # Make predictions
        y_pred = optimizer.predict(test_embeddings)
        y_true = test_labels.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred.cpu().numpy())
    
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
    
    # Prepare inputs
    inputs, _ = prepare_batch_for_training(log_sequences, vocab)
    inputs = inputs.to(device)
    
    # Get encodings
    with torch.no_grad():
        embeddings = encoder(inputs)
    
    # Calculate anomaly scores
    scores = []
    for embedding in embeddings:
        # Calculate distance to normal templates in the lookup
        embedding_np = embedding.cpu().numpy()
        distances = []
        
        for template_id, template_embedding in template_lookup.items():
            distance = np.linalg.norm(embedding_np - template_embedding)
            distances.append(distance)
        
        # Use minimum distance as anomaly score
        if distances:
            min_distance = min(distances)
            scores.append(min_distance)
        else:
            scores.append(float('inf'))  # If no templates in lookup
    
    # Make binary predictions
    predictions = [1 if score > threshold else 0 for score in scores]
    
    return predictions, scores 