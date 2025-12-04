#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Episode sampler for multi-domain meta-learning
"""

from typing import Dict, List, Iterator, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from core.entities.domains import DomainBatch
from utils.data_processing import prepare_batch_for_training


class InstanceDataset(Dataset):
    """PyTorch Dataset wrapper for list of instances."""
    
    def __init__(self, instances, vocab):
        self.instances = instances
        self.vocab = vocab
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]


def sample_batch_from_list(instances: list, n_samples: int, vocab, device=None) -> tuple:
    """Sample n_samples from a list of instances and convert to model input format.
    
    Args:
        instances: List of instance objects
        n_samples: Number of samples to collect
        vocab: Vocabulary for encoding
        device: Device to move tensors to (optional)
        
    Returns:
        (x_tuple, y_tensor): x_tuple is (words, masks, lengths), y is labels
    """
    import random
    
    # Sample randomly if we have more than needed
    if len(instances) > n_samples:
        sampled = random.sample(instances, n_samples)
    else:
        sampled = instances
    
    if not sampled:
        raise ValueError(f"Could not collect {n_samples} samples from list")
    
    # Convert to tensors using existing utility
    tinst, labels = prepare_batch_for_training(sampled, vocab, verbose=False)
    
    # Prepare model inputs format: (words, masks, lengths)
    words = tinst
    masks = torch.ones_like(words, dtype=torch.float)
    lengths = torch.sum(masks, dim=1)
    
    if device is not None:
        words = words.to(device)
        masks = masks.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
    
    # Return tuple format expected by AttGRUModel
    x = (words, masks, lengths)
    y = labels
    
    return x, y


def sample_episode(
    domain_data: Dict[str, dict],
    target_data: dict,
    n_support_src: int,
    n_query_src: int,
    n_support_tgt: int,
    n_query_tgt: int,
    vocab,
    device: Optional[torch.device] = None,
) -> List[DomainBatch]:
    """Sample an episode containing support/query sets from all domains.
    
    Args:
        domain_data: Dictionary mapping domain IDs to {"support": list, "query": list}
        target_data: Dictionary with "support" and "query" lists for target
        n_support_src: Number of support samples per source domain
        n_query_src: Number of query samples per source domain
        n_support_tgt: Number of support samples for target domain
        n_query_tgt: Number of query samples for target domain
        vocab: Vocabulary for encoding instances
        device: Device to move tensors to (optional)
        
    Returns:
        List of DomainBatch objects, one per domain (sources + target)
    """
    domain_batches = []
    
    # Sample from each source domain
    for domain_id, data in domain_data.items():
        support_list = data.get("support", [])
        query_list = data.get("query", [])
        
        try:
            # Sample from support and query lists
            supp_x, supp_y = sample_batch_from_list(support_list, n_support_src, vocab, device)
            query_x, query_y = sample_batch_from_list(query_list, n_query_src, vocab, device)
            
            domain_batch = DomainBatch(
                domain_id=domain_id,
                support_x=supp_x,  # Tuple (words, masks, lengths)
                support_y=supp_y,
                query_x=query_x,   # Tuple (words, masks, lengths)
                query_y=query_y,
            )
            
            domain_batches.append(domain_batch)
        except (ValueError, IndexError) as e:
            # Skip this domain if we can't get enough samples
            print(f"Warning: Could not sample episode for domain {domain_id}: {e}")
            continue
    
    # Sample from target domain
    target_support = target_data.get("support", [])
    target_query = target_data.get("query", [])
    
    try:
        tgt_supp_x, tgt_supp_y = sample_batch_from_list(target_support, n_support_tgt, vocab, device)
        tgt_query_x, tgt_query_y = sample_batch_from_list(target_query, n_query_tgt, vocab, device)
        
        target_batch = DomainBatch(
            domain_id="TARGET",
            support_x=tgt_supp_x,
            support_y=tgt_supp_y,
            query_x=tgt_query_x,
            query_y=tgt_query_y,
        )
        
        domain_batches.append(target_batch)
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not sample episode for TARGET domain: {e}")
    
    return domain_batches

