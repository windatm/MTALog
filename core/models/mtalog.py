#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MTALog: Multi-source meta-learning model with source weighting
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.entities.domains import DomainBatch
from core.models.gru import AttGRUModel
from core.meta.prototypes import compute_prototypes
from core.meta.source_weighting import compute_source_scores, scores_to_alphas


class MTALog(nn.Module):
    """Multi-source meta-learning model with source weighting α_s.
    
    This model wraps an encoder (AttGRUModel) and implements episode-based
    training with source reliability and distance-based weighting.
    """
    
    def __init__(
        self,
        encoder: AttGRUModel,
        lambda_align: float = 1.0,
        lambda_cl: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Initialize MTALog model.
        
        Args:
            encoder: AttGRUModel encoder (already includes classifier)
            lambda_align: Weight for alignment loss
            lambda_cl: Weight for contrastive loss
            optimizer: Optimizer for parameter updates (optional, can be set later)
        """
        super().__init__()
        self.encoder = encoder
        self.lambda_align = lambda_align
        self.lambda_cl = lambda_cl
        self.optimizer = optimizer
        
        # Classifier is already part of AttGRUModel
        # We can access it via encoder.classifier
    
    def encode_batch(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of inputs to latent representations.
        
        Args:
            inputs: Tuple of (words, masks, lengths) tensors
                - words: [N, seq_len] word IDs
                - masks: [N, seq_len] attention masks
                - lengths: [N] sequence lengths
                
        Returns:
            z: [N, d] latent representations
        """
        # AttGRUModel.forward returns (logits, latent, lengths)
        _, latent, _ = self.encoder(inputs)
        return latent
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            inputs: Tuple of (words, masks, lengths)
            
        Returns:
            (logits, latent): Classification logits and latent representations
        """
        return self.encoder(inputs)
    
    def train_on_episode(
        self,
        domain_batches: List[DomainBatch],
        w_r: float = 1.0,
        w_D: float = 1.0,
        beta_margin: float = 0.1,
        kappa: float = 1.0,
        source_weighting_mode: str = "reliability_plus_distance",
    ) -> Dict[str, float]:
        """Train on a single episode with source weighting.
        
        Args:
            domain_batches: List of DomainBatch objects for all domains
            w_r: Weight for reliability in source scoring
            w_D: Weight for distance in source scoring
            beta_margin: Weight for margin in reliability
            kappa: Temperature for softmax in α_s computation
            source_weighting_mode: "none", "distance_only", or "reliability_plus_distance"
            
        Returns:
            Dictionary with loss values and metrics
        """
        self.train()
        
        # 1. Compute prototypes for all domains
        prototypes = compute_prototypes(domain_batches, self)
        
        # 2. Compute source scores & alphas
        scores = compute_source_scores(
            prototypes=prototypes,
            domain_batches=domain_batches,
            model=self,
            w_r=w_r,
            w_D=w_D,
            beta_margin=beta_margin,
            mode=source_weighting_mode,
        )
        alphas = scores_to_alphas(scores, kappa=kappa)
        
        # 3. Compute L_cls: weighted classification loss
        L_cls = 0.0
        total_weight = 0.0
        
        for db in domain_batches:
            # Encode query set
            z_query = self.encode_batch(db.query_x)  # [N_q, d]
            
            # Get logits from classifier
            logits, _, _ = self.encoder(db.query_x)  # [N_q, 2]
            
            # Compute cross-entropy loss
            ce = F.cross_entropy(logits, db.query_y, reduction="none")  # [N_q]
            
            # Get domain weight
            if db.domain_id == "TARGET":
                w_domain = 1.0  # Fixed weight for target
            else:
                w_domain = alphas.get(db.domain_id, 0.0)
            
            L_cls = L_cls + w_domain * ce.sum()
            total_weight += w_domain * ce.numel()
        
        if total_weight > 0:
            L_cls = L_cls / total_weight
        else:
            L_cls = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 4. Compute L_align: prototype alignment loss
        # Align target normal prototype with weighted average of source normal prototypes
        L_align = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if 0 in prototypes.get("TARGET", {}):
            mu_T_0 = prototypes["TARGET"][0]
            src_mu0_weighted = None
            norm_alpha = 0.0
            
            for db in domain_batches:
                if db.domain_id == "TARGET":
                    continue
                s = db.domain_id
                if 0 not in prototypes.get(s, {}):
                    continue
                
                alpha_s = alphas.get(s, 0.0)
                if alpha_s > 0:
                    if src_mu0_weighted is None:
                        src_mu0_weighted = alpha_s * prototypes[s][0]
                    else:
                        src_mu0_weighted = src_mu0_weighted + alpha_s * prototypes[s][0]
                    norm_alpha += alpha_s
            
            if src_mu0_weighted is not None and norm_alpha > 0:
                src_mu0_weighted = src_mu0_weighted / norm_alpha
                L_align = torch.dist(mu_T_0, src_mu0_weighted, p=2) ** 2
        
        # 5. Compute L_CL: contrastive loss (placeholder for now)
        L_CL = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Total loss
        L_total = L_cls + self.lambda_align * L_align + self.lambda_cl * L_CL
        
        # Backward and update
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            L_total.backward()
            self.optimizer.step()
        
        return {
            "L_total": float(L_total.item()),
            "L_cls": float(L_cls.item()),
            "L_align": float(L_align.item()),
            "L_CL": float(L_CL.item()),
            "alphas": alphas,
            "scores": scores,
        }

