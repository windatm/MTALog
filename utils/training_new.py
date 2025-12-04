#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
New training functions using MetaTrainer
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List

from constants import DEVICE
from core.models.gru import AttGRUModel
from core.models.mtalog import MTALog
from core.meta.trainer import MetaTrainer
from core.meta.episode_sampler import InstanceDataset
from core.module.Optimizer import Optimizer


class ListDataset(Dataset):
    """Simple dataset wrapper for list of instances."""
    
    def __init__(self, instances):
        self.instances = instances
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]


def create_dataloader_from_list(instances: List, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from a list of instances.
    
    Args:
        instances: List of instance objects
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    dataset = ListDataset(instances)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(params, logger, source_data, target_data):
    """Train model using new episode-based meta-learning.
    
    Args:
        params: Configuration parameters
        logger: Logger instance
        source_data: Dictionary with source system data
        target_data: Dictionary with target system data
    """
    logger.info("=== Starting episode-based meta-learning training ===")
    
    # Get target encoder
    target_encoder = target_data["target_encoder"]
    target_vocab = target_data.get("target_vocab") or target_encoder.vocab
    
    # Create MTALog model
    model = MTALog(
        encoder=target_encoder,
        lambda_align=params.get("lambda_align", 1.0),
        lambda_cl=params.get("lambda_cl", 1.0),
    )
    
    # Create optimizer
    optimizer = Optimizer(
        parameter=model.parameters(),
        lr=params.get("gamma", 8e-3)
    )
    model.optimizer = optimizer
    
    # Prepare domain data dictionaries
    domain_data = {}
    for source_system in params["source_systems"]:
        support_set = source_data["source_support_sets"][source_system]
        query_set = source_data["source_query_sets"][source_system]
        domain_data[source_system] = {
            "support": support_set,
            "query": query_set,
        }
    
    # Prepare target data
    target_support_set = target_data["support_set"]
    target_query_set = target_data["query_set"]
    target_data_dict = {
        "support": target_support_set,
        "query": target_query_set,
    }
    
    # Create MetaTrainer
    trainer = MetaTrainer(
        model=model,
        device=DEVICE,
        logger=logger,
        log_interval=params.get("log_interval", 10),
    )
    
    # Training parameters
    n_support_src = params.get("n_support_src", 16)
    n_query_src = params.get("n_query_src", 16)
    n_support_tgt = params.get("n_support_tgt", 16)
    n_query_tgt = params.get("n_query_tgt", 16)
    
    # Number of episodes = num_epochs * episodes_per_epoch
    num_episodes = params.get("num_episodes", params.get("num_epochs", 5) * 100)
    
    # Train
    history = trainer.train(
        num_episodes=num_episodes,
        domain_data=domain_data,
        target_data=target_data_dict,
        n_support_src=n_support_src,
        n_query_src=n_query_src,
        n_support_tgt=n_support_tgt,
        n_query_tgt=n_query_tgt,
        vocab=target_vocab,
        w_r=params.get("w_r", 1.0),
        w_D=params.get("w_D", 1.0),
        beta_margin=params.get("beta_margin", 0.1),
        kappa=params.get("kappa", 1.0),
        source_weighting_mode=params.get("source_weighting_mode", "reliability_plus_distance"),
    )
    
    logger.info("Training completed")
    logger.info(f"Final losses - L_total: {history['L_total'][-1]:.4f}, "
                f"L_cls: {history['L_cls'][-1]:.4f}, "
                f"L_align: {history['L_align'][-1]:.4f}")
    
    return model, history

