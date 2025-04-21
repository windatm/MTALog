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

def prepare_batch_for_training(logs, vocab, max_length=100):
    """Prepare a batch of logs for training"""
    # Convert log templates to indices
    sequences = []
    labels = []
    
    for log in logs:
        # Convert template IDs to vocab indices
        sequence = [vocab.template_to_idx(template) for template in log.template_ids]
        
        # Pad or truncate to max_length
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence + [0] * (max_length - len(sequence))  # Pad with zeros
        
        sequences.append(sequence)
        labels.append(log.label)
    
    return torch.tensor(sequences), torch.tensor(labels)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    # True positives, false positives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    } 