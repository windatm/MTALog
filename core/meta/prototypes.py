#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prototype computation for multi-domain meta-learning
"""

from typing import Dict, List
import torch

from core.entities.domains import DomainBatch
from core.models.mtalog import MTALog


def compute_prototypes(
    domain_batches: List[DomainBatch],
    model: MTALog,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute prototypes (class centroids) for each domain.
    
    Args:
        domain_batches: List of DomainBatch objects for all domains
        model: MTALog model for encoding
        
    Returns:
        prototypes: Dictionary mapping domain_id -> {class_label -> prototype_tensor}
            Example: {"BGL": {0: μ_BGL^0, 1: μ_BGL^1}, "TARGET": {0: μ_T^0, 1: μ_T^1}}
    """
    prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
    
    model.eval()
    with torch.no_grad():
        for db in domain_batches:
            # Encode support set
            # db.support_x is a tuple (words, masks, lengths)
            z_supp = model.encode_batch(db.support_x)  # [N_supp, d]
            
            prot_per_class: Dict[int, torch.Tensor] = {}
            
            # Compute prototype for each class (0: normal, 1: anomaly)
            for y_val in (0, 1):
                mask = (db.support_y == y_val)
                if mask.sum() == 0:
                    # No samples of this class in support set
                    continue
                
                # Compute mean embedding for this class
                mu = z_supp[mask].mean(dim=0)  # [d]
                prot_per_class[int(y_val)] = mu
            
            prototypes[db.domain_id] = prot_per_class
    
    return prototypes

