#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for MTALog
"""

import torch
import numpy as np

def batch_to_tensor(batch, device):
    """Convert a batch of data to tensors on the specified device"""
    inputs = torch.tensor(batch, device=device)
    return inputs


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