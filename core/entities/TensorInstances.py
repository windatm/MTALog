#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tensor-based instance classes for model training and inference
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class TInstWithLogits:
    """Class representing tensor instances with logits for model training"""
    
    batch_size: int
    sequence_length: int
    tag_size: int
    
    # Tensor attributes
    src_ids: List[int] = field(default_factory=list)
    src_words: Tensor = field(init=False)
    src_masks: Tensor = field(init=False)
    tags: Tensor = field(init=False)
    g_truth: Tensor = field(init=False)
    word_len: Tensor = field(init=False)
    
    # Cache for inputs
    _inputs_cache: Optional[Tuple[Tensor, Tensor, Tensor]] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize tensors with zeros"""
        self.src_words = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.long)
        self.src_masks = torch.zeros(self.batch_size, self.sequence_length)
        self.tags = torch.zeros(self.batch_size, self.tag_size)
        self.g_truth = torch.zeros(self.batch_size, dtype=torch.long)
        self.word_len = torch.zeros(self.batch_size, dtype=torch.long)
    
    def to(self, device: Union[str, torch.device]) -> 'TInstWithLogits':
        """Move all tensors to the specified device"""
        self.src_words = self.src_words.to(device)
        self.src_masks = self.src_masks.to(device)
        self.tags = self.tags.to(device)
        self.g_truth = self.g_truth.to(device)
        self.word_len = self.word_len.to(device)
        self._inputs_cache = None
        return self
    
    @property
    def inputs(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return a tuple of tensors needed for model input"""
        if self._inputs_cache is not None:
            return self._inputs_cache
            
        inputs_tuple = (self.src_words, self.src_masks, self.word_len)
        self._inputs_cache = inputs_tuple
        return inputs_tuple
    
    @inputs.setter
    def inputs(self, value: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Set the inputs tuple directly"""
        self.src_words, self.src_masks, self.word_len = value[:3]
        self._inputs_cache = value
    
    @property
    def ids(self) -> List[int]:
        """Return source IDs"""
        return self.src_ids
    
    @property
    def targets(self) -> Tensor:
        """Return target tensors"""
        return self.tags
    
    @property
    def truth(self) -> Tensor:
        """Return ground truth tensors"""
        return self.g_truth


@dataclass
class TInstWithoutLogits:
    """Class representing tensor instances without logits for model inference"""
    
    batch_size: int
    sequence_length: int
    tag_size: int
    
    # Tensor attributes
    src_words: Tensor = field(init=False)
    src_masks: Tensor = field(init=False)
    tags: Tensor = field(init=False)
    word_len: Tensor = field(init=False)
    
    def __post_init__(self):
        """Initialize tensors with zeros"""
        self.src_words = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.long)
        self.src_masks = torch.zeros(self.batch_size, self.sequence_length)
        self.tags = torch.zeros(self.batch_size, dtype=torch.long)
        self.word_len = torch.zeros(self.batch_size, dtype=torch.long)
    
    def to(self, device: Union[str, torch.device]) -> 'TInstWithoutLogits':
        """Move all tensors to the specified device"""
        self.src_words = self.src_words.to(device)
        self.src_masks = self.src_masks.to(device)
        self.tags = self.tags.to(device)
        self.word_len = self.word_len.to(device)
        return self
    
    @property
    def inputs(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return a tuple of tensors needed for model input"""
        return self.src_words, self.src_masks, self.word_len
    
    @property
    def targets(self) -> Tensor:
        """Return target tensors"""
        return self.tags
    
    @property
    def truth(self) -> Tensor:
        """Return ground truth tensors"""
        return self.tags
