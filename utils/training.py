#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and evaluation functions for MTALog
"""

import torch
import numpy as np
import random
from collections import defaultdict
import traceback
import os

from utils.data_processing import (
    prepare_batch_for_training
)


def create_batches(data, batch_size):
    """
    Split a dataset into batches of specified size.
    
    Args:
        data: List of data instances
        batch_size: Size of each batch
        
    Returns:
        list: List of batches, where each batch is a list of instances
    """
    batches = []
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batches.append(data[start_idx:end_idx])
    
    return batches


def ensure_vocab_has_template_to_idx(vocab):
    """Add a template_to_idx method to the vocab object if it doesn't have one."""
    if not hasattr(vocab, 'template_to_idx'):
        # Add the method dynamically
        vocab.template_to_idx = lambda template: vocab.word2id(str(template))
    return vocab


# Old loss functions removed - will be replaced by new episode-based training


def classify_logs(query_embeddings, centroid, support_embeddings, z_score=2.0):
    """
    Phân loại logs dựa trên khoảng cách đến centroid của normal cluster
    
    Args:
        query_embeddings: Embeddings cần phân loại
        centroid: Tâm điểm của normal cluster
        support_embeddings: Embeddings từ support set (normal logs)
        z_score: Số lần độ lệch chuẩn để xác định ngưỡng
        
    Returns:
        predictions: Dự đoán nhãn (0: normal, 1: anomaly)
        confidence: Độ tin cậy của dự đoán (khoảng cách đã chuẩn hóa)
        threshold: Ngưỡng phân loại đã được sử dụng
    """
    # Tính khoảng cách từ query embeddings đến centroid
    query_distances = torch.norm(query_embeddings - centroid, dim=1)
    
    # Tính khoảng cách từ support embeddings đến centroid
    support_distances = torch.norm(support_embeddings - centroid, dim=1)
    
    # Tính ngưỡng dựa trên phân phối khoảng cách của support set
    mean_distance = torch.mean(support_distances)
    std_distance = torch.std(support_distances)
    
    # Ngưỡng = mean + z_score * std 
    threshold = mean_distance + z_score * std_distance
    
    # Phân loại: 0 (normal) nếu khoảng cách <= threshold, 1 (anomaly) nếu không
    predictions = (query_distances > threshold).int()
    
    # Confidence scores: khoảng cách được chuẩn hóa
    confidence = query_distances / threshold
    
    return predictions, confidence, threshold


# Old meta_train_step removed - replaced by episode-based training in MTALog model


def meta_test_step(support_batch, query_batch, encoder, optimizer=None, device='cpu', batch_size=32, margin=0.5, logger=None):
    """
    Perform a meta-testing step using the trained model
    
    Args:
        support_batch: Support set batch (normal samples)
        query_batch: Query set batch (mix of normal and abnormal samples)
        encoder: Neural network encoder model
        optimizer: Optimizer for parameter updates (can be None for evaluation)
        device: Device to run computations on
        batch_size: Batch size for training
        margin: Margin for contrastive loss
        logger: Logger for tracking process
    
    Returns:
        loss: Loss value for this step
        metrics: Dictionary containing evaluation metrics
    """
    encoder.eval()  # Set model to evaluation mode
    
    metrics = {}
    if not query_batch:
        if logger:
            logger.warning("Empty query batch in meta_test_step")
        return 0.0, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, 
                     "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    # Log label distribution if logger is provided
    if logger:
        # Kiểm tra chi tiết nhãn để debug
        label_types = {}
        label_values = {}
        for inst in query_batch:
            if hasattr(inst, 'label'):
                label = inst.label
                label_type = type(label).__name__
                
                # Đếm theo loại
                if label_type not in label_types:
                    label_types[label_type] = 0
                label_types[label_type] += 1
                
                # Đếm theo giá trị
                label_str = str(label)
                if label_str not in label_values:
                    label_values[label_str] = 0
                label_values[label_str] += 1
        
        if label_types:
            logger.debug(f"Query batch label types: {label_types}")
        if label_values:
            logger.debug(f"Query batch label values: {label_values}")
        
        # Xác định nhãn dạng numeric
        normal_count = sum(1 for inst in query_batch if hasattr(inst, 'label') and 
                          (inst.label == 0 or 
                          (isinstance(inst.label, str) and 
                           inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))
        anomaly_count = sum(1 for inst in query_batch if hasattr(inst, 'label') and 
                           (inst.label == 1 or 
                           (isinstance(inst.label, str) and 
                            inst.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos'])))
        
        logger.debug(f"Query batch: {normal_count} normal, {anomaly_count} anomalous")
        
        if normal_count == 0 and anomaly_count == 0 and query_batch:
            logger.warning(f"No recognized labels found in query batch of size {len(query_batch)}. Check label format.")
            # Kiểm tra xem có nhãn nào nhưng không được nhận diện không
            if label_values:
                logger.warning(f"Found unrecognized label values: {label_values}. Might need to update label normalization.")
        
        if support_batch:
            # Tương tự cho support batch
            support_normal = sum(1 for inst in support_batch if hasattr(inst, 'label') and 
                                (inst.label == 0 or 
                                (isinstance(inst.label, str) and 
                                 inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))
            
            support_anomaly = sum(1 for inst in support_batch if hasattr(inst, 'label') and 
                                 (inst.label == 1 or 
                                 (isinstance(inst.label, str) and 
                                  inst.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos'])))
            
            logger.debug(f"Support batch: {support_normal} normal, {support_anomaly} anomalous")
            
            if support_normal == 0 and support_anomaly == 0 and support_batch:
                # Thử kiểm tra label đầu tiên để debug
                if hasattr(support_batch[0], 'label'):
                    logger.warning(f"Support batch has zero recognized labels. First instance label: {support_batch[0].label} (type: {type(support_batch[0].label).__name__})")
                else:
                    logger.warning("Support batch instances don't have 'label' attribute")
    
    # Process query batch
    query_tinst, query_labels = prepare_batch_for_training(query_batch, encoder.vocab, verbose=False)
    query_words = query_tinst.to(device)
    query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
    query_word_len = torch.sum(query_masks, dim=1).to(device)
    query_model_inputs = (query_words, query_masks, query_word_len)
    
    if query_model_inputs is None:
        if logger:
            logger.warning("Invalid query batch tensor instances in meta_test_step")
        return 0.0, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, 
                     "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    query_logits, _, query_embeddings = encoder(query_model_inputs)
    
    # Get normal centroid - either use stored one or compute from support set
    if hasattr(encoder, 'normal_centroid') and encoder.normal_centroid is not None:
        normal_centroid = encoder.normal_centroid
        if logger:
            logger.debug("Using pre-computed normal centroid from encoder")
    elif support_batch:
        # Filter support set to only include normal instances - FIX: Handle string labels
        normal_support = [inst for inst in support_batch if 
                         inst.label == 0 or 
                         (isinstance(inst.label, str) and 
                          inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])]
        
        if logger:
            logger.debug(f"Filtered {len(normal_support)}/{len(support_batch)} normal instances for centroid calculation")
        
        if not normal_support:
            if logger:
                logger.warning("No normal instances in support batch for centroid calculation")
            # Use mean of query embeddings as fallback (not ideal but prevents errors)
            normal_centroid = torch.mean(query_embeddings, dim=0, keepdim=True)
        else:
            # Compute normal centroid from support set
            support_tinst, _ = prepare_batch_for_training(normal_support, encoder.vocab, verbose=False)
            support_words = support_tinst.to(device)
            support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
            support_word_len = torch.sum(support_masks, dim=1).to(device)
            support_model_inputs = (support_words, support_masks, support_word_len)
            
            if support_model_inputs is None:
                if logger:
                    logger.warning("Invalid support batch tensor instances in meta_test_step")
                # Use mean of query embeddings as fallback
                normal_centroid = torch.mean(query_embeddings, dim=0, keepdim=True)
            else:
                with torch.no_grad():
                    _, _, support_embeddings = encoder(support_model_inputs)
                normal_centroid = torch.mean(support_embeddings, dim=0, keepdim=True)
    else:
        # If no support batch and no stored centroid, use mean of query embeddings as fallback
        if logger:
            logger.warning("No support batch or stored centroid, using query mean as centroid")
        normal_centroid = torch.mean(query_embeddings, dim=0, keepdim=True)
    
    # Calculate distances to normal centroid
    with torch.no_grad():
        # Calculate distances from each query embedding to the normal centroid
        distances = torch.norm(query_embeddings - normal_centroid, dim=1)
        
        # Predict anomalies (1) for instances with distance > threshold, normal (0) otherwise
        threshold = torch.mean(distances) + margin * torch.std(distances)
        predicted_labels = (distances > threshold).long()
        
        # Get true labels from query batch - handle string labels
        numeric_labels = []
        for inst in query_batch:
            if isinstance(inst.label, str):
                if inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg']:
                    numeric_labels.append(0)
                elif inst.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos']:
                    numeric_labels.append(1)
                else:
                    # Try to convert to integer if possible
                    try:
                        numeric_labels.append(int(inst.label))
                    except ValueError:
                        # Default to 0 for unknown strings
                        numeric_labels.append(0)
            else:
                # Use the label directly if it's not a string
                numeric_labels.append(int(inst.label))
        
        true_labels = torch.tensor(numeric_labels, device=device)
        
        # Calculate metrics
        tp = torch.sum((predicted_labels == 1) & (true_labels == 1)).item()
        fp = torch.sum((predicted_labels == 1) & (true_labels == 0)).item()
        fn = torch.sum((predicted_labels == 0) & (true_labels == 1)).item()
        tn = torch.sum((predicted_labels == 0) & (true_labels == 0)).item()
        
        # Avoid division by zero
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    
    # Calculate loss if we need to perform optimization
    if optimizer is not None:
        encoder.train()
        optimizer.zero_grad()
        
        # Recompute embeddings in train mode
        query_logits, _, query_embeddings = encoder(query_model_inputs)
        
        # Simple mean squared error loss for distances
        mse_loss = torch.nn.MSELoss()
        
        # Expected distances: small for normal, large for anomalies
        expected_distances = torch.zeros_like(distances)
        for i, inst in enumerate(query_batch):
            # Handle normal vs anomaly
            is_normal = (inst.label == 0 or 
                        (isinstance(inst.label, str) and 
                         inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg']))
            
            if is_normal:  # Normal
                expected_distances[i] = 0.0  # Should be close to centroid
            else:  # Anomaly
                expected_distances[i] = margin  # Should be at least 'margin' away
        
        # Calculate distances again
        distances = torch.norm(query_embeddings - normal_centroid, dim=1)
        
        # Calculate loss
        loss = mse_loss(distances, expected_distances)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        return loss.item(), metrics
    
    # If no optimization, return 0 loss
    return 0.0, metrics


def log_centroid_changes(current_centroid, previous_centroid, logger, epoch=None):
    """
    Log changes in centroid between epochs
    
    Args:
        current_centroid: Current centroid tensor
        previous_centroid: Previous centroid tensor from last epoch (can be None for first epoch)
        logger: Logger instance (can be None)
        epoch: Current epoch number (optional)
    """
    # Get basic centroid stats
    centroid_norm = torch.norm(current_centroid).item()
    centroid_mean = torch.mean(current_centroid).item()
    centroid_std = torch.std(current_centroid).item()
    centroid_min = torch.min(current_centroid).item()
    centroid_max = torch.max(current_centroid).item()
    
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    print(f"{epoch_str}Centroid stats - Norm: {centroid_norm:.4f}, Mean: {centroid_mean:.4f}, "
          f"Std: {centroid_std:.4f}, Min: {centroid_min:.4f}, Max: {centroid_max:.4f}")
    
    # Always log important summary of centroid 
    print(f"{epoch_str}Centroid summary - Norm: {centroid_norm:.4f}, Mean: {centroid_mean:.4f}")
    
    # Compare with previous centroid if available
    if previous_centroid is not None:
        # Calculate cosine similarity
        current_flat = current_centroid.view(-1)
        previous_flat = previous_centroid.view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(current_flat, previous_flat, dim=0).item()
        
        # Calculate Euclidean distance
        euclid_dist = torch.norm(current_centroid - previous_centroid).item()
        
        # Calculate relative change in norm
        prev_norm = torch.norm(previous_centroid).item()
        norm_change = (centroid_norm - prev_norm) / prev_norm if prev_norm > 0 else float('inf')
        
        print(f"{epoch_str}Centroid change - Cosine sim: {cos_sim:.4f}, "
              f"Euclidean dist: {euclid_dist:.4f}, Norm change: {norm_change:.4f}")
        
        # Always log important summary of changes
        print(f"{epoch_str}Centroid change summary - Cosine sim: {cos_sim:.4f}, Norm change: {norm_change:.2%}")


# Old train_model and train_model_single removed - replaced by MetaTrainer in core/meta/trainer.py


def evaluate_model(target_support_set, target_query_set, encoder, device, batch_size=32, logger=None, threshold=None):
    """
    Evaluate a trained model on the target domain data
    
    Args:
        target_support_set: Support set from target domain (normal samples)
        target_query_set: Query set from target domain (mix of normal and anomalous)
        encoder: Neural network encoder model
        device: Device to run computations on
        batch_size: Batch size for evaluation
        logger: Optional logger for debug information
        threshold: Optional fixed threshold for anomaly detection
        
    Returns:
        Dictionary with evaluation metrics
    """
    if logger:
        logger.info(f"Evaluating model on {len(target_query_set)} query instances using {len(target_support_set)} support instances")
    
    # Validation to prevent empty sets
    if not target_query_set:
        if logger:
            logger.warning("Empty query set in evaluate_model. Cannot evaluate.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, 
                "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    # Validate labels in the query set
    if logger:
        # Thêm kiểm tra chi tiết về nhãn dữ liệu
        logger.info(f"Performing detailed label inspection for query set...")
        label_types = {}
        label_values = {}
        
        for i, inst in enumerate(target_query_set[:min(100, len(target_query_set))]):  # Kiểm tra 100 instance đầu tiên
            label = getattr(inst, 'label', None)
            label_type = type(label).__name__
            
            # Đếm theo loại
            if label_type not in label_types:
                label_types[label_type] = 0
            label_types[label_type] += 1
            
            # Đếm theo giá trị
            label_str = str(label)
            if label_str not in label_values:
                label_values[label_str] = 0
            label_values[label_str] += 1
            
            # In ra chi tiết một số instance mẫu
            if i < 5:  # In 5 instance đầu tiên
                logger.info(f"Sample instance {i}: label={label} (type={label_type}), dir(inst)={dir(inst)}")
        
        logger.info(f"Label types distribution: {label_types}")
        logger.info(f"Label values distribution: {label_values}")
        
        # Kiểm tra xem có các attribute khác có thể chứa nhãn không
        potential_label_attrs = ['label', 'anomaly', 'is_anomaly', 'class', 'category', 'tag']
        for attr in potential_label_attrs:
            if any(hasattr(inst, attr) for inst in target_query_set[:100]):
                logger.info(f"Found alternative label attribute: '{attr}'")
        
        normal_count = sum(1 for inst in target_query_set if hasattr(inst, 'label') and 
                           (inst.label == 0 or 
                            (isinstance(inst.label, str) and 
                             inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))
        anomaly_count = sum(1 for inst in target_query_set if hasattr(inst, 'label') and 
                           (inst.label == 1 or 
                            (isinstance(inst.label, str) and 
                             inst.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos'])))
        
        logger.info(f"Query set label distribution: {normal_count} normal, {anomaly_count} anomalous")
        
        if normal_count == 0 and anomaly_count == 0:
            logger.warning("Query set has no valid labels. Check data preprocessing.")
    
    # Set model to evaluation mode
    encoder.eval()
    
    # Process in batches to handle large datasets
    num_batches = max(1, len(target_query_set) // batch_size)
    metrics_sum = defaultdict(float)
    valid_batches = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(target_query_set))
        query_batch = target_query_set[start_idx:end_idx]
        
        # Skip empty batches
        if not query_batch:
            if logger:
                logger.warning(f"Skipping empty batch {i+1}/{num_batches}")
            continue
        
        # Use a subset of support set for efficiency if it's large
        support_batch = target_support_set[:min(len(target_support_set), batch_size * 2)]
        
        if logger:
            normal_count = sum(1 for inst in query_batch if hasattr(inst, 'label') and 
                           (inst.label == 0 or 
                            (isinstance(inst.label, str) and 
                             inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))
            anomaly_count = sum(1 for inst in query_batch if hasattr(inst, 'label') and 
                            (inst.label == 1 or 
                             (isinstance(inst.label, str) and 
                              inst.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos'])))
            
            logger.debug(f"Batch {i+1}/{num_batches}: Query batch has {normal_count} normal, {anomaly_count} anomalous")
            
            # Skip batches with no valid labels
            if normal_count == 0 and anomaly_count == 0:
                logger.warning(f"Skipping batch {i+1}/{num_batches} with no valid labels")
                continue
        
        try:
            # Run evaluation step
            _, batch_metrics = meta_test_step(
                support_batch=support_batch, 
                query_batch=query_batch,
                encoder=encoder,
                device=device,
                batch_size=batch_size,
                logger=logger
            )
            
            # Add batch metrics to running totals
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
                
            valid_batches += 1
                
        except Exception as e:
            if logger:
                logger.error(f"Error in evaluation batch {i+1}/{num_batches}: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Calculate average metrics (avoid division by zero)
    if valid_batches > 0:
        metrics_avg = {k: v / valid_batches for k, v in metrics_sum.items()}
    else:
        if logger:
            logger.warning("No valid batches were processed during evaluation")
        print("Warning: No valid batches were processed during evaluation")
        metrics_avg = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, 
                      "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    # In kết quả đánh giá
    print(f"Evaluation metrics: "
           f"Accuracy={metrics_avg['accuracy']:.4f}, "
           f"Precision={metrics_avg['precision']:.4f}, "
           f"Recall={metrics_avg['recall']:.4f}, "
           f"F1={metrics_avg['f1']:.4f}")
    
    if logger:
        logger.info(f"Evaluation metrics: "
                   f"Accuracy={metrics_avg['accuracy']:.4f}, "
                   f"Precision={metrics_avg['precision']:.4f}, "
                   f"Recall={metrics_avg['recall']:.4f}, "
                   f"F1={metrics_avg['f1']:.4f}")
        
        # Log centroid and threshold information during evaluation
        for i, (support_batch, query_batch) in enumerate(zip(
                [target_support_set[:batch_size]],
                [target_query_set[:batch_size]]
            )):
            if support_batch:
                # Filter normal instances for centroid calculation
                normal_support = [inst for inst in support_batch if 
                                 (hasattr(inst, 'label') and (inst.label == 0 or 
                                 (isinstance(inst.label, str) and inst.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))]
                
                if normal_support:
                    if hasattr(encoder, 'vocab'):
                        vocab = encoder.vocab
                        # Compute centroid from support set
                        support_tinst, _ = prepare_batch_for_training(normal_support, vocab, verbose=False)
                        support_words = support_tinst.to(device)
                        support_masks = torch.ones_like(support_words, dtype=torch.float, device=device)
                        support_word_len = torch.sum(support_masks, dim=1).to(device)
                        support_model_inputs = (support_words, support_masks, support_word_len)
                        
                        with torch.no_grad():
                            _, _, support_embeddings = encoder(support_model_inputs)
                            
                        eval_centroid = torch.mean(support_embeddings, dim=0, keepdim=True)
                        
                        # Log centroid statistics
                        centroid_norm = torch.norm(eval_centroid).item()
                        centroid_mean = torch.mean(eval_centroid).item()
                        centroid_std = torch.std(eval_centroid).item()
                        centroid_min = torch.min(eval_centroid).item()
                        centroid_max = torch.max(eval_centroid).item()
                        
                        logger.info(f"Evaluation centroid - Norm: {centroid_norm:.4f}, Mean: {centroid_mean:.4f}, "
                                   f"Std: {centroid_std:.4f}, Min: {centroid_min:.4f}, Max: {centroid_max:.4f}")
                        
                        # Optionally log distances for debugging separation
                        if query_batch:
                            query_tinst, _ = prepare_batch_for_training(query_batch, vocab, verbose=False)
                            query_words = query_tinst.to(device)
                            query_masks = torch.ones_like(query_words, dtype=torch.float, device=device)
                            query_word_len = torch.sum(query_masks, dim=1).to(device)
                            query_model_inputs = (query_words, query_masks, query_word_len)
                            
                            with torch.no_grad():
                                _, _, query_embeddings = encoder(query_model_inputs)
                                distances = torch.norm(query_embeddings - eval_centroid, dim=1)
                                
                            # Log distance statistics
                            logger.info(f"Distance stats - Mean: {torch.mean(distances).item():.4f}, "
                                       f"Std: {torch.std(distances).item():.4f}, "
                                       f"Min: {torch.min(distances).item():.4f}, "
                                       f"Max: {torch.max(distances).item():.4f}")
                break  # Only process the first batch
        
    return metrics_avg


def predict(
    log_sequences,
    encoder,
    template_lookup,
    device,
    threshold=0.5,
    logger=None
):
    """
    Make predictions for new log sequences using a memory-efficient approach
    
    Args:
        log_sequences: List of log sequences to classify
        encoder: Trained encoder model
        template_lookup: Lookup table for template embeddings
        device: Device to run computations on
        threshold: Classification threshold
        logger: Optional logger for debug information
        
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
    
    # Thêm kiểm tra và thông báo về dữ liệu
    if len(log_sequences) > 0 and logger:
        logger.info(f"Predict: Processing {len(log_sequences)} log sequences in {num_chunks} chunks")
        # Kiểm tra label distribution
        if all(hasattr(log, 'label') for log in log_sequences[:min(100, len(log_sequences))]):
            normal_count = sum(1 for log in log_sequences if 
                              hasattr(log, 'label') and (log.label == 0 or 
                                                       (isinstance(log.label, str) and 
                                                        log.label.lower() in ['normal', 'negative', '0', 'norm', 'neg'])))
            anomaly_count = sum(1 for log in log_sequences if 
                               hasattr(log, 'label') and (log.label == 1 or 
                                                        (isinstance(log.label, str) and 
                                                         log.label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos'])))
            logger.info(f"Predict: Data has {normal_count} normal, {anomaly_count} anomalous instances (based on 'label' attribute)")
        
        # Kiểm tra sequence attribute
        sequence_lens = [len(getattr(log, 'sequence', [])) for log in log_sequences[:min(10, len(log_sequences))]]
        if sequence_lens:
            logger.info(f"Predict: First 10 sequence lengths: {sequence_lens}")
    
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