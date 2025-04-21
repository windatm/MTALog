#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for MTALog
"""

import torch
import numpy as np
from tqdm import tqdm

def batch_to_tensor(batch, device):
    """Convert a batch of data to tensors on the specified device"""
    inputs = torch.tensor(batch, device=device)
    return inputs

def encode_batch(batch, encoder, device):
    """Encode a batch of sequences using the provided encoder"""
    # Move data to device
    inputs = batch_to_tensor(batch, device)
    
    # Forward pass through encoder
    with torch.no_grad():
        outputs = encoder(inputs)
    
    return outputs

def encode_sequences(sequences, encoder, batch_size, device):
    """Encode a list of sequences using the provided encoder in batches"""
    # Create batches
    num_sequences = len(sequences)
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    # Process each batch
    encoded_sequences = []
    for i in tqdm(range(num_batches), desc="Encoding sequences"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_sequences)
        batch = sequences[start_idx:end_idx]
        
        # Encode batch
        encoded_batch = encode_batch(batch, encoder, device)
        encoded_sequences.append(encoded_batch)
    
    # Concatenate results
    all_encoded = torch.cat(encoded_sequences, dim=0)
    return all_encoded

def create_embedding_lookup(templates, encoder, device):
    """Create a lookup table from template IDs to their embeddings"""
    lookup = {}
    unique_templates = list(set(templates))
    
    # Create batches to avoid memory issues
    batch_size = 128
    num_templates = len(unique_templates)
    num_batches = (num_templates + batch_size - 1) // batch_size
    
    # Process each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_templates)
        batch = unique_templates[start_idx:end_idx]
        
        # Encode batch
        batch_tensor = torch.tensor(batch, device=device).unsqueeze(1)  # Add sequence dim
        with torch.no_grad():
            encodings = encoder(batch_tensor).squeeze(1)  # Remove sequence dim
        
        # Add to lookup
        for j, template_id in enumerate(batch):
            lookup[template_id] = encodings[j].cpu().numpy()
    
    return lookup

def calculate_similarity(query_embedding, support_embeddings):
    """Calculate cosine similarity between query and support embeddings"""
    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm
    
    similarities = []
    for support_embedding in support_embeddings:
        support_norm = np.linalg.norm(support_embedding)
        if support_norm > 0:
            support_embedding = support_embedding / support_norm
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, support_embedding)
        similarities.append(similarity)
    
    return similarities

def find_nearest_templates(query_embedding, template_lookup, k=5):
    """Find k nearest templates to the query embedding"""
    similarities = []
    for template_id, embedding in template_lookup.items():
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((template_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:k]

def aggregate_sequence_embeddings(sequence_embeddings):
    """Aggregate embeddings from a sequence into a single representation"""
    # Simple mean pooling
    return np.mean(sequence_embeddings, axis=0)

def prepare_batch_for_training(logs, vocab, max_length=100, verbose=False):
    """
    Prepare a batch of logs for training with enhanced label handling
    
    Args:
        logs: List of log instances
        vocab: Vocabulary object
        max_length: Maximum sequence length
        verbose: Whether to print verbose information
        
    Returns:
        tuple: (sequences tensor, labels tensor)
    """
    # Convert log templates to indices
    sequences = []
    labels = []
    
    # Input validation
    if logs is None or len(logs) == 0:
        if verbose:
            print("Warning: Empty log batch provided to prepare_batch_for_training")
        return torch.zeros((1, max_length), dtype=torch.long), torch.zeros(1, dtype=torch.long)
    
    if vocab is None:
        if verbose:
            print("Warning: No vocab provided to prepare_batch_for_training")
        return torch.zeros((len(logs), max_length), dtype=torch.long), torch.zeros(len(logs), dtype=torch.long)
    
    # Print diagnostics about the first log
    if verbose:
        first_log = logs[0]
        print(f"Log type: {type(first_log).__name__}")
        print(f"Log attributes: {dir(first_log)}")
        print(f"Has template_ids: {hasattr(first_log, 'template_ids')}")
        print(f"Has sequence: {hasattr(first_log, 'sequence')}")
        if hasattr(first_log, 'sequence'):
            print(f"Sequence type: {type(first_log.sequence).__name__}")
            print(f"Sequence length: {len(first_log.sequence)}")
            if len(first_log.sequence) > 0:
                print(f"First sequence item: {first_log.sequence[0]}")
        
        # Print vocab info
        print(f"Vocab type: {type(vocab).__name__}")
        print(f"Has template_to_idx: {hasattr(vocab, 'template_to_idx')}")
        print(f"Has word2id: {hasattr(vocab, 'word2id')}")
    
    # Counter for label distribution
    label_counter = {'normal': 0, 'anomaly': 0, 'unknown': 0}
    
    for log in logs:
        # Get sequence from log
        if hasattr(log, 'template_ids'):
            # Use template_ids if available
            template_sequence = log.template_ids
        elif hasattr(log, 'sequence'):
            # Fall back to using sequence attribute
            template_sequence = log.sequence
        else:
            # If no sequence information is available, use an empty sequence
            if verbose:
                print(f"Warning: Log ID {getattr(log, 'id', 'unknown')} has no template_ids or sequence")
            template_sequence = []
        
        # Convert to vocab indices - use template_to_idx if available, otherwise word2id
        try:
            if hasattr(vocab, 'template_to_idx'):
                sequence = [vocab.template_to_idx(template) for template in template_sequence]
            else:
                # Fall back to word2id method if template_to_idx doesn't exist
                sequence = [vocab.word2id(str(template)) for template in template_sequence]
        except Exception as e:
            if verbose:
                print(f"Error converting templates to indices: {str(e)}")
            # Use default indices (0) if conversion fails
            sequence = [0] * min(len(template_sequence), max_length)
        
        # Pad or truncate to max_length
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence + [0] * (max_length - len(sequence))  # Pad with zeros
        
        sequences.append(sequence)
        
        # Get label with extensive normalization
        if hasattr(log, 'label'):
            log_label = log.label
            
            # Handle various label formats
            if isinstance(log_label, str):
                # Normalize string labels
                if log_label.lower() in ['normal', 'negative', '0', 'norm', 'neg']:
                    normalized_label = 0
                    label_counter['normal'] += 1
                elif log_label.lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos']:
                    normalized_label = 1
                    label_counter['anomaly'] += 1
                elif log_label.isdigit():
                    normalized_label = int(log_label)
                    label_counter['normal' if normalized_label == 0 else 'anomaly'] += 1
                else:
                    # Default to 0 for unknown strings
                    normalized_label = 0
                    label_counter['unknown'] += 1
                    if verbose:
                        print(f"Unknown label string '{log_label}' for log ID {getattr(log, 'id', 'unknown')}, defaulting to 0")
            elif isinstance(log_label, (int, float, bool, np.integer, np.floating)):
                # Normalize numeric labels
                normalized_label = 1 if log_label > 0.5 or log_label is True else 0
                label_counter['normal' if normalized_label == 0 else 'anomaly'] += 1
            else:
                # Default to 0 for unknown types
                normalized_label = 0
                label_counter['unknown'] += 1
                if verbose:
                    print(f"Unknown label type {type(log_label)} for log ID {getattr(log, 'id', 'unknown')}, defaulting to 0")
            
            labels.append(normalized_label)
        else:
            # Default to normal (0) if no label is available
            labels.append(0)
            label_counter['unknown'] += 1
            if verbose:
                print(f"Warning: Log ID {getattr(log, 'id', 'unknown')} has no label")
    
    if verbose:
        print(f"Prepared {len(sequences)} sequences with length {max_length}")
        print(f"Label distribution: {label_counter}")
        
    # Convert to tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Check if we have a balanced set
    if verbose and label_counter['normal'] > 0 and label_counter['anomaly'] > 0:
        normal_ratio = label_counter['normal'] / (label_counter['normal'] + label_counter['anomaly'])
        print(f"Class balance: {normal_ratio:.2f} normal, {1-normal_ratio:.2f} anomaly")
    
    return sequences_tensor, labels_tensor

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics with robust handling of edge cases
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        
    Returns:
        dict: Dictionary of metrics including accuracy, precision, recall, F1, and confusion matrix values
    """
    # Input validation
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0
        }
    
    # Convert to numpy arrays if not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle common errors - mismatch in shapes
    if y_true.shape != y_pred.shape:
        print(f"Warning: Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}")
        # Truncate to the shorter length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Check if all predictions are the same class
    all_same_pred = (np.unique(y_pred).size <= 1)
    all_same_true = (np.unique(y_true).size <= 1)
    
    # If all predictions are the same and all true values are the same
    if all_same_pred and all_same_true:
        # If they match (all correct), use perfect metrics
        if np.array_equal(y_pred, y_true):
            accuracy = 1.0
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            # If they don't match (all wrong), use zero metrics
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        # Calculate metrics with careful handling of edge cases
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    } 