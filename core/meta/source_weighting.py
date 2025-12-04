#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source weighting computation: reliability r_s, distance D_s, and weights α_s
"""

from typing import Dict, List, Optional
import torch

from core.entities.domains import DomainBatch
from core.models.mtalog import MTALog


def compute_source_scores(
    prototypes: Dict[str, Dict[int, torch.Tensor]],
    domain_batches: List[DomainBatch],
    model: MTALog,
    w_r: float = 1.0,
    w_D: float = 1.0,
    beta_margin: float = 0.1,
    mode: str = "reliability_plus_distance",
) -> Dict[str, float]:
    """Compute source scores based on reliability and distance.
    
    Args:
        prototypes: Dictionary mapping domain_id -> {class_label -> prototype}
        domain_batches: List of DomainBatch objects
        model: MTALog model for encoding
        w_r: Weight for reliability component
        w_D: Weight for distance component
        beta_margin: Weight for margin in reliability computation
        mode: Scoring mode - "none", "distance_only", or "reliability_plus_distance"
        
    Returns:
        scores: Dictionary mapping source domain_id -> score_s
    """
    # Find TARGET domain batch
    target_db = None
    for db in domain_batches:
        if db.domain_id == "TARGET":
            target_db = db
            break
    
    if target_db is None:
        raise ValueError("TARGET domain not found in domain_batches.")
    
    # Encode target support set
    model.eval()
    with torch.no_grad():
        z_T_supp = model.encode_batch(target_db.support_x)  # [N_T, d]
        y_T_supp = target_db.support_y  # [N_T]
    
    scores: Dict[str, float] = {}
    prot_T = prototypes.get("TARGET", {})
    
    # Iterate over source domains (exclude TARGET)
    for db in domain_batches:
        if db.domain_id == "TARGET":
            continue
        
        s = db.domain_id
        prot_s = prototypes.get(s, {})
        
        if mode == "none":
            # Uniform scores
            scores[s] = 0.0
            continue
        
        # Compute D_s: prototype distance between source and target
        D_s = 0.0
        num_classes = 0
        for y_val in (0, 1):
            if y_val in prot_s and y_val in prot_T:
                dist = torch.dist(prot_s[y_val], prot_T[y_val], p=2).item()
                D_s += dist ** 2
                num_classes += 1
        
        if mode == "distance_only":
            # Score based only on distance (closer = better)
            scores[s] = -D_s if num_classes > 0 else -1e6
            continue
        
        # Mode: reliability_plus_distance
        # Compute r_s: reliability based on nearest-prototype prediction
        preds = []
        margins = []
        
        for i in range(z_T_supp.size(0)):
            z_i = z_T_supp[i]
            
            # Get distances to source prototypes
            d0 = torch.dist(z_i, prot_s[0], p=2).item() if 0 in prot_s else None
            d1 = torch.dist(z_i, prot_s[1], p=2).item() if 1 in prot_s else None
            
            if d0 is None or d1 is None:
                # Missing prototype - penalize this source
                true_y = int(y_T_supp[i].item())
                preds.append(1 - true_y)  # Wrong prediction
                margins.append(0.0)
                continue
            
            # Predict based on nearest prototype
            pred_y = 0 if d0 < d1 else 1
            preds.append(pred_y)
            
            # Compute margin
            true_y = int(y_T_supp[i].item())
            d_true = d0 if true_y == 0 else d1
            d_wrong = d1 if true_y == 0 else d0
            margin = (d_wrong - d_true)
            margins.append(margin)
        
        if len(preds) == 0:
            scores[s] = -1e6
            continue
        
        # Compute accuracy and average margin
        preds_t = torch.tensor(preds, device=y_T_supp.device, dtype=torch.long)
        acc_s = (preds_t == y_T_supp).float().mean().item()
        margin_s = float(sum(margins) / len(margins)) if margins else 0.0
        
        # Reliability score: accuracy + margin component
        r_s = acc_s + beta_margin * margin_s
        
        # Combined score: reliability - distance
        score_s = w_r * r_s - w_D * D_s
        scores[s] = score_s
    
    return scores


def scores_to_alphas(scores: Dict[str, float], kappa: float = 1.0) -> Dict[str, float]:
    """Convert source scores to normalized weights α_s using softmax.
    
    Args:
        scores: Dictionary mapping domain_id -> score
        kappa: Temperature parameter for softmax (higher = sharper distribution)
        
    Returns:
        alphas: Dictionary mapping domain_id -> α_s (normalized weights)
    """
    if not scores:
        return {}
    
    keys = list(scores.keys())
    vals = torch.tensor([scores[k] for k in keys])
    
    # Apply softmax with temperature
    alphas_tensor = torch.softmax(kappa * vals, dim=0)
    
    return {k: float(v) for k, v in zip(keys, alphas_tensor)}

