#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DomainBatch: Data structure for multi-domain episode batches
"""

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class DomainBatch:
    """Represents a batch of support and query data for a single domain in an episode.
    
    Attributes:
        domain_id: Identifier for the domain (e.g., "BGL", "HDFS", "TARGET")
        support_x: Support set features - tuple of (words, masks, lengths) for AttGRUModel
        support_y: Support set labels [N_supp]
        query_x: Query set features - tuple of (words, masks, lengths) for AttGRUModel
        query_y: Query set labels [N_query]
    """
    domain_id: str
    support_x: tuple  # Tuple of (words, masks, lengths) tensors
    support_y: torch.Tensor   # [N_supp]
    query_x: tuple     # Tuple of (words, masks, lengths) tensors
    query_y: torch.Tensor     # [N_query]
    
    def to(self, device):
        """Move all tensors to the specified device."""
        # support_x and query_x are tuples, need to move each element
        self.support_x = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in self.support_x)
        self.support_y = self.support_y.to(device)
        self.query_x = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in self.query_x)
        self.query_y = self.query_y.to(device)
        return self

