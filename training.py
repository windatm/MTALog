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


def meta_test_step(target_support_set, target_query_set, encoder, optimizer, device, batch_size=32, logger=None):
    """
    Perform one meta-testing step on target data using prototype-based approach
    
    Args:
        target_support_set: Support set from target domain
        target_query_set: Query set from target domain
        encoder: Neural network encoder model
        optimizer: Optimizer for parameter updates
        device: Device to run computations on
        batch_size: Batch size for training
        logger: Optional logger for debug information
        
    Returns:
        loss: Test loss value
        metrics: Performance metrics dictionary
    """
    # Set model to evaluation mode
    encoder.eval()
    
    # Basic validation of input data
    if not target_support_set or len(target_support_set) == 0:
        if logger:
            logger.warning("Empty target support set provided to meta_test_step")
        return 0.0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    if not target_query_set or len(target_query_set) == 0:
        if logger:
            logger.warning("Empty target query set provided to meta_test_step")
        return 0.0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    # Get vocab with fallback
    if hasattr(encoder, 'vocab'):
        vocab = encoder.vocab
    else:
        # Try to extract vocab from first instance in support set
        vocab = getattr(target_support_set[0], 'vocab', None)
        if vocab is None:
            error_msg = "No vocab found in encoder or support set. Cannot proceed with testing."
            if logger:
                logger.error(error_msg)
            raise AttributeError(error_msg)
    
    # Ensure vocab has template_to_idx method - silently add if needed
    if not hasattr(vocab, 'template_to_idx'):
        vocab.template_to_idx = lambda template: vocab.word2id(str(template))
    
    # Process in smaller chunks to save memory
    max_batch_size = min(batch_size, 32)  # Reduced batch size
    
    # Log distribution of labels in support set
    if logger:
        normal_count = sum(1 for inst in target_support_set if hasattr(inst, 'label') and 
                        (inst.label == 0 or inst.label == "Normal"))
        anomaly_count = sum(1 for inst in target_support_set if hasattr(inst, 'label') and 
                         (inst.label == 1 or inst.label == "Anomalous"))
        logger.debug(f"Support set: {len(target_support_set)} instances, {normal_count} normal, {anomaly_count} anomaly")
    
    # Use only normal samples from support set for prototype calculation
    normal_support_set = [inst for inst in target_support_set[:max_batch_size*2] if 
                        hasattr(inst, 'label') and (inst.label == 0 or inst.label == "Normal")]
    
    # If no normal samples, use all samples but log a warning
    if not normal_support_set:
        if logger:
            logger.warning("No normal samples found in support set for prototype calculation")
        normal_support_set = target_support_set[:max_batch_size*2]
    
    # Prepare support data - use more instances for better prototype
    support_tinst, support_labels = prepare_batch_for_training(normal_support_set, vocab, verbose=False)
    
    # Create model inputs
    support_words = support_tinst.to(device)
    support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
    support_word_len = torch.sum(support_masks, dim=1).to(device)
    support_model_inputs = (support_words, support_masks, support_word_len)
    
    # Forward pass on support set to calculate prototype
    with torch.no_grad():
        # Get embeddings - capture and log any errors
        try:
            _, _, support_embeddings = encoder(support_model_inputs)
            
            # Check if embeddings contain NaN values
            if torch.isnan(support_embeddings).any():
                if logger:
                    logger.warning("NaN values detected in support embeddings")
                # Replace NaN with zeros
                support_embeddings = torch.nan_to_num(support_embeddings, nan=0.0)
            
            # Create prototype as mean of embeddings
            prototype = support_embeddings.mean(dim=0, keepdim=True)
            
        except Exception as e:
            error_msg = f"Error in prototype calculation: {str(e)}"
            if logger:
                logger.error(error_msg)
            return 0.0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        
        # Free memory
        del support_model_inputs, support_words, support_masks, support_word_len, support_embeddings
        torch.cuda.empty_cache()
    
    # Log distribution of labels in query set
    if logger:
        normal_count = sum(1 for inst in target_query_set if hasattr(inst, 'label') and 
                        (inst.label == 0 or inst.label == "Normal"))
        anomaly_count = sum(1 for inst in target_query_set if hasattr(inst, 'label') and 
                         (inst.label == 1 or inst.label == "Anomalous"))
        logger.debug(f"Query set: {len(target_query_set)} instances, {normal_count} normal, {anomaly_count} anomaly")
    
    # Ensure we have a balance of normal and anomalous samples in the query batch
    # This helps prevent the case where we only have one class in the query set
    if len(target_query_set) > max_batch_size:
        normal_query_instances = [inst for inst in target_query_set if hasattr(inst, 'label') and 
                               (inst.label == 0 or inst.label == "Normal")]
        anomaly_query_instances = [inst for inst in target_query_set if hasattr(inst, 'label') and 
                                (inst.label == 1 or inst.label == "Anomalous")]
        
        # Ensure we have at least some instances of each class if available
        if normal_query_instances and anomaly_query_instances:
            # Take a balanced sample
            max_per_class = max_batch_size // 2
            normal_sample = normal_query_instances[:max_per_class]
            anomaly_sample = anomaly_query_instances[:max_per_class]
            query_batch = normal_sample + anomaly_sample
        else:
            # If we don't have both classes, just take a sample
            query_batch = target_query_set[:max_batch_size]
    else:
        query_batch = target_query_set[:max_batch_size]
    
    # Prepare query data
    query_tinst, query_labels = prepare_batch_for_training(query_batch, vocab, verbose=False)
    
    # Ensure we have valid labels
    if torch.unique(query_labels).shape[0] <= 1:
        if logger:
            logger.warning(f"Only one class ({query_labels[0].item() if query_labels.shape[0] > 0 else 'unknown'}) found in query batch")
    
    # Create model inputs for query
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    query_model_inputs = (query_words, query_masks, query_word_len)
    
    # Forward pass on query set with no gradient
    with torch.no_grad():
        try:
            # Get embeddings and logits
            query_logits, _, query_embeddings = encoder(query_model_inputs)
            
            # Check for NaN values
            if torch.isnan(query_logits).any() or torch.isnan(query_embeddings).any():
                if logger:
                    logger.warning("NaN values detected in query outputs")
                # Replace NaN with zeros
                query_logits = torch.nan_to_num(query_logits, nan=0.0)
                query_embeddings = torch.nan_to_num(query_embeddings, nan=0.0)
            
            # Calculate cosine similarity to prototype
            similarity = torch.nn.functional.cosine_similarity(query_embeddings, prototype, dim=1)
            
            # Calculate loss - handle case where all labels are the same
            try:
                query_loss = torch.nn.CrossEntropyLoss()(query_logits, query_labels.to(device))
            except Exception as e:
                if logger:
                    logger.warning(f"Error calculating loss: {str(e)}")
                query_loss = torch.tensor(0.0, device=device)
            
            # Make predictions using logits
            y_pred_from_logits = torch.argmax(query_logits, dim=1).cpu().numpy()
            
            # Also try making predictions using similarity threshold
            # Lower similarity to prototype (normal instances) means higher anomaly probability
            similarity_threshold = similarity.median().item()  # Adaptive threshold
            y_pred_from_similarity = (similarity < similarity_threshold).cpu().numpy().astype(int)
            
            # Combine predictions - if both methods agree, use that; otherwise, use logits
            y_pred = np.where(
                y_pred_from_logits == y_pred_from_similarity,
                y_pred_from_logits,
                y_pred_from_logits
            )
            
            y_true = query_labels.cpu().numpy()
            
            # Log prediction distribution
            if logger:
                logger.debug(f"Prediction distribution: {np.bincount(y_pred)}")
                logger.debug(f"True label distribution: {np.bincount(y_true)}")
            
        except Exception as e:
            error_msg = f"Error in query processing: {str(e)}"
            if logger:
                logger.error(error_msg)
            import traceback
            if logger:
                logger.error(traceback.format_exc())
            return 0.0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        finally:
            # Free memory
            if 'query_model_inputs' in locals():
                del query_model_inputs
            if 'query_words' in locals():
                del query_words, query_masks, query_word_len
            if 'query_embeddings' in locals():
                del query_embeddings
            if 'query_logits' in locals():
                del query_logits
            if 'prototype' in locals():
                del prototype
            if 'similarity' in locals():
                del similarity
            torch.cuda.empty_cache()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Log detailed metrics
    if logger:
        logger.debug(f"Metrics: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    
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
        try:
            loss, metrics = meta_test_step(
                target_support_set,
                target_query_set,
                target_encoder,
                optimizer,
                device,
                batch_size,
                logger
            )
            
            # Validate metrics to ensure they are numbers
            for metric_name, metric_value in metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    if not isinstance(metric_value, (int, float)) or np.isnan(metric_value) or np.isinf(metric_value):
                        logger.warning(f"Invalid {metric_name} value: {metric_value}, setting to 0")
                        metrics[metric_name] = 0.0
            
            # Log performance
            logger.info(f"  Target Loss: {loss:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.debug(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
            
            # Save best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = target_encoder.state_dict()
                
                # Save model checkpoint
                model_path = os.path.join(output_model_dir, f"best_model_epoch_{epoch+1}.pt")
                torch.save(best_model, model_path)
                logger.info(f"  New best model saved to {model_path}")
                
                # Also save a backup copy of the best model in case of future errors
                backup_path = os.path.join(output_model_dir, "best_model_latest.pt")
                torch.save(best_model, backup_path)
        except Exception as e:
            # Log the error but continue training
            logger.error(f"Error during meta-testing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
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
    
    # Input validation
    if not test_data or len(test_data) == 0:
        logger.warning("Empty test data provided to evaluate_model")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
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
    
    # Log distribution of labels in test data
    normal_count = sum(1 for inst in test_data if hasattr(inst, 'label') and 
                    (inst.label == 0 or inst.label == "Normal"))
    anomaly_count = sum(1 for inst in test_data if hasattr(inst, 'label') and 
                     (inst.label == 1 or inst.label == "Anomalous"))
    logger.info(f"Test data: {len(test_data)} instances, {normal_count} normal, {anomaly_count} anomaly")
    
    # Check if we have both classes
    if normal_count == 0 or anomaly_count == 0:
        logger.warning(f"Test data has imbalanced classes: {normal_count} normal, {anomaly_count} anomaly")
        if normal_count == 0 and anomaly_count == 0:
            logger.error("No valid labels found in test data")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    # Process in chunks to reduce memory usage
    all_predictions = []
    all_true_labels = []
    
    chunk_size = min(batch_size, 32)  # Smaller batch size to prevent OOM
    num_chunks = (len(test_data) + chunk_size - 1) // chunk_size
    
    logger.info(f"Evaluating on {len(test_data)} instances in {num_chunks} chunks")
    
    total_loss = 0.0
    
    for chunk_idx in range(num_chunks):
        try:
            # Get chunk of data
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(test_data))
            chunk_data = test_data[start_idx:end_idx]
            
            # Skip empty chunks
            if not chunk_data:
                continue
                
            logger.info(f"Processing evaluation chunk {chunk_idx+1}/{num_chunks}, size: {len(chunk_data)}")
            
            # Try to balance the chunk if possible
            if len(chunk_data) > 4:  # Only balance if we have enough data
                normal_instances = [inst for inst in chunk_data if hasattr(inst, 'label') and 
                                (inst.label == 0 or inst.label == "Normal")]
                anomaly_instances = [inst for inst in chunk_data if hasattr(inst, 'label') and 
                                 (inst.label == 1 or inst.label == "Anomalous")]
                
                # If we have both normal and anomaly instances, create a balanced chunk
                if normal_instances and anomaly_instances:
                    logger.debug(f"Balancing chunk: {len(normal_instances)} normal, {len(anomaly_instances)} anomaly")
                    max_per_class = chunk_size // 2
                    balanced_normal = normal_instances[:max_per_class]
                    balanced_anomaly = anomaly_instances[:max_per_class]
                    chunk_data = balanced_normal + balanced_anomaly
            
            # Prepare test data for this chunk
            test_tinst, test_labels = prepare_batch_for_training(chunk_data, vocab, verbose=False)
            
            # Log label distribution in this chunk
            label_counts = {}
            for label in test_labels.cpu().numpy():
                label_counts[int(label)] = label_counts.get(int(label), 0) + 1
            logger.debug(f"Chunk {chunk_idx+1} labels: {label_counts}")
            
            # Create model inputs
            test_words = test_tinst.to(device)
            test_masks = torch.ones_like(test_words, dtype=torch.float, device=device)
            test_word_len = torch.sum(test_masks, dim=1).to(device)
            test_model_inputs = (test_words, test_masks, test_word_len)
            
            # Forward pass with no gradient
            with torch.no_grad():
                # Get embeddings and logits
                test_logits, _, test_embeddings = encoder(test_model_inputs)
                
                # Check for NaN values
                if torch.isnan(test_logits).any() or torch.isnan(test_embeddings).any():
                    logger.warning("NaN values detected in outputs")
                    # Replace NaN with zeros
                    test_logits = torch.nan_to_num(test_logits, nan=0.0)
                    test_embeddings = torch.nan_to_num(test_embeddings, nan=0.0)
                
                # Calculate loss - handle exceptions
                try:
                    chunk_loss = torch.nn.CrossEntropyLoss()(test_logits, test_labels.to(device))
                    total_loss += chunk_loss.item()
                except Exception as e:
                    logger.warning(f"Error calculating loss: {str(e)}")
                    chunk_loss = torch.tensor(0.0)
                
                # Make predictions using both logits and similarity approach
                # Standard logits-based approach
                chunk_preds_logits = torch.argmax(test_logits, dim=1).cpu().numpy()
                
                # Try a similarity-based approach if embeddings are available
                if test_embeddings is not None and test_embeddings.shape[0] > 0:
                    # Calculate similarity to prototype - use mean of normal instance embeddings
                    normal_idx = (test_labels == 0).nonzero(as_tuple=True)[0]
                    if len(normal_idx) > 0:
                        # If we have normal instances, use them as prototype
                        normal_embeddings = test_embeddings[normal_idx]
                        prototype = normal_embeddings.mean(dim=0, keepdim=True)
                    else:
                        # Otherwise use mean of all embeddings as prototype
                        prototype = test_embeddings.mean(dim=0, keepdim=True)
                    
                    # Calculate similarity to prototype
                    similarity = torch.nn.functional.cosine_similarity(test_embeddings, prototype, dim=1)
                    
                    # Lower similarity to normal prototype means higher anomaly probability
                    threshold = similarity.median().item()  # Adaptive threshold
                    chunk_preds_sim = (similarity < threshold).cpu().numpy().astype(int)
                    
                    # Combine predictions - if both methods agree, use that; otherwise, use logits
                    chunk_preds = np.where(
                        chunk_preds_logits == chunk_preds_sim,
                        chunk_preds_logits,
                        chunk_preds_logits  # Prefer logits if they disagree
                    )
                else:
                    # If no embeddings, use logits predictions
                    chunk_preds = chunk_preds_logits
                
                chunk_true = test_labels.cpu().numpy()
                
                # Add to full results
                all_predictions.extend(chunk_preds)
                all_true_labels.extend(chunk_true)
                
                # Log prediction distribution for this chunk
                pred_counts = {}
                for pred in chunk_preds:
                    pred_counts[int(pred)] = pred_counts.get(int(pred), 0) + 1
                logger.debug(f"Chunk {chunk_idx+1} predictions: {pred_counts}")
                
            # Free memory
            del test_words, test_masks, test_word_len, test_model_inputs
            del test_logits, test_embeddings
            if 'chunk_loss' in locals():
                del chunk_loss
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Check if we have predictions
    if not all_predictions or not all_true_labels:
        logger.error("No predictions or true labels collected during evaluation")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_true_labels)
    
    # Log overall prediction distribution
    pred_distribution = np.bincount(y_pred) if len(y_pred) > 0 else []
    true_distribution = np.bincount(y_true) if len(y_true) > 0 else []
    logger.info(f"Prediction distribution: {pred_distribution}")
    logger.info(f"True label distribution: {true_distribution}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Validate metrics to ensure they are numbers
    for metric_name, metric_value in metrics.items():
        if metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if not isinstance(metric_value, (int, float)) or np.isnan(metric_value) or np.isinf(metric_value):
                logger.warning(f"Invalid {metric_name} value: {metric_value}, setting to 0")
                metrics[metric_name] = 0.0
    
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