#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MetaTrainer: Training loop for multi-source meta-learning
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import logging

from core.entities.domains import DomainBatch
from core.models.mtalog import MTALog
from core.meta.episode_sampler import sample_episode


class MetaTrainer:
    """Trainer for multi-source meta-learning with source weighting."""
    
    def __init__(
        self,
        model: MTALog,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ):
        """Initialize MetaTrainer.
        
        Args:
            model: MTALog model to train
            device: Device to run training on
            logger: Logger instance (optional)
            log_interval: Log metrics every N episodes
        """
        self.model = model
        self.device = device
        self.logger = logger
        self.log_interval = log_interval
        
        # Move model to device
        self.model.to(device)
    
    def train(
        self,
        num_episodes: int,
        domain_data: Dict[str, dict],
        target_data: dict,
        n_support_src: int,
        n_query_src: int,
        n_support_tgt: int,
        n_query_tgt: int,
        vocab,
        w_r: float = 1.0,
        w_D: float = 1.0,
        beta_margin: float = 0.1,
        kappa: float = 1.0,
        source_weighting_mode: str = "reliability_plus_distance",
    ) -> Dict[str, List[float]]:
        """Train the model for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            domain_data: Dictionary mapping source domain IDs to {"support": list, "query": list}
            target_data: Dictionary with "support" and "query" lists for target
            n_support_src: Support samples per source domain
            n_query_src: Query samples per source domain
            n_support_tgt: Support samples for target
            n_query_tgt: Query samples for target
            vocab: Vocabulary for encoding
            w_r: Weight for reliability
            w_D: Weight for distance
            beta_margin: Margin weight in reliability
            kappa: Softmax temperature
            source_weighting_mode: Scoring mode
            
        Returns:
            Dictionary with training history (losses, metrics, etc.)
        """
        history = {
            "L_total": [],
            "L_cls": [],
            "L_align": [],
            "L_CL": [],
            "alphas_history": [],
        }
        
        if self.logger:
            self.logger.info(f"Starting training for {num_episodes} episodes")
            self.logger.info(f"Source weighting mode: {source_weighting_mode}")
        
        for episode_idx in range(num_episodes):
            try:
                # Sample episode
                domain_batches = sample_episode(
                    domain_data=domain_data,
                    target_data=target_data,
                    n_support_src=n_support_src,
                    n_query_src=n_query_src,
                    n_support_tgt=n_support_tgt,
                    n_query_tgt=n_query_tgt,
                    vocab=vocab,
                    device=self.device,
                )
                
                # Train on episode
                loss_dict = self.model.train_on_episode(
                    domain_batches=domain_batches,
                    w_r=w_r,
                    w_D=w_D,
                    beta_margin=beta_margin,
                    kappa=kappa,
                    source_weighting_mode=source_weighting_mode,
                )
                
                # Record history
                history["L_total"].append(loss_dict["L_total"])
                history["L_cls"].append(loss_dict["L_cls"])
                history["L_align"].append(loss_dict["L_align"])
                history["L_CL"].append(loss_dict["L_CL"])
                history["alphas_history"].append(loss_dict.get("alphas", {}))
                
                # Logging
                if (episode_idx + 1) % self.log_interval == 0:
                    msg = (
                        f"Episode {episode_idx + 1}/{num_episodes} - "
                        f"L_total: {loss_dict['L_total']:.4f}, "
                        f"L_cls: {loss_dict['L_cls']:.4f}, "
                        f"L_align: {loss_dict['L_align']:.4f}, "
                        f"L_CL: {loss_dict['L_CL']:.4f}"
                    )
                    
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                    
                    # Log alphas
                    alphas = loss_dict.get("alphas", {})
                    if alphas:
                        alpha_str = ", ".join([f"{k}: {v:.3f}" for k, v in alphas.items()])
                        alpha_msg = f"  Alphas: {alpha_str}"
                        if self.logger:
                            self.logger.info(alpha_msg)
                        else:
                            print(alpha_msg)
                    
                    # Log scores if available
                    scores = loss_dict.get("scores", {})
                    if scores:
                        score_str = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items()])
                        score_msg = f"  Scores: {score_str}"
                        if self.logger:
                            self.logger.info(score_msg)
                        else:
                            print(score_msg)
            
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in episode {episode_idx + 1}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                else:
                    print(f"Error in episode {episode_idx + 1}: {e}")
                continue
        
        if self.logger:
            self.logger.info("Training completed")
        
        return history

