#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AttGRUModel: Attention-based GRU for log anomaly detection.
Uses bidirectional GRU with attention mechanism for sequence-level classification.
"""

from pathlib import Path
from typing import Tuple
import logging

import torch
import torch.nn as nn
from torch import Tensor

from constants import DEVICE, LOG_ROOT, SESSION
from module.Attention import LinearAttention
from module.Common import NonLinear
from module.CPUEmbedding import CPUEmbedding


class AttGRUModel(nn.Module):
    """Attention-based GRU model for log anomaly detection.
    
    Architecture:
    1. Static embeddings - 2. Bidirectional GRU - 3. Attention - 4. Classification
    """
    
    def __init__(
        self, 
        vocab,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Core parameters
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.sent_dim = 2 * hidden_size  # bidirectional
        
        # Model components
        self.word_embed = self._setup_embeddings()
        self.rnn = nn.GRU(
            input_size=vocab.word_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = self._setup_attention()
        self.classifier = NonLinear(self.sent_dim, 2)
        
        # Utilities
        self.dropout = nn.Dropout(dropout)
        self.logger = self._setup_logger()
            
        self.to(DEVICE)

    def _setup_embeddings(self) -> CPUEmbedding:
        """Initialize static embeddings."""
        embed = CPUEmbedding(
            self.vocab.vocab_size,
            self.vocab.word_dim,
            padding_idx=self.vocab.vocab_size - 1
        )
        embed.weight.data.copy_(torch.from_numpy(self.vocab.embeddings))
        embed.weight.requires_grad = False
        return embed

    def _setup_attention(self) -> LinearAttention:
        """Initialize attention mechanism."""
        guide = nn.Parameter(torch.randn(self.sent_dim))
        self.register_parameter('attention_guide', guide)
        return LinearAttention(self.sent_dim, self.sent_dim)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("AttGRU")
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s")
        
        # Console output
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        
        # File output
        log_file = Path(LOG_ROOT) / "AttGRU.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Logs: {log_file}")
        return logger

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Process input sequences through the model.
        
        Args:
            inputs: (word_ids, attention_mask, sequence_lengths)
            
        Returns:
            (logits, latent_repr): Classification logits and latent representations
        """
        # Prepare inputs
        words, mask, lengths = (x.to(next(self.parameters()).device) for x in inputs)
        
        # Embedding
        x = self.word_embed(words)
        x = self.dropout(x) if self.training else x
        
        # GRU encoding
        hidden_states, _ = self.rnn(x)
        
        # Attention
        batch_size = x.size(0)
        guide = self.attention_guide.unsqueeze(1).expand(-1, batch_size).t()
        weights = self.attention(guide, hidden_states, mask)
        weights = weights.view(batch_size, hidden_states.size(1), -1)
        
        # Final representations
        latent = (hidden_states * weights).sum(dim=1)
        logits = self.classifier(latent)
        
        return logits, latent, lengths

    def to(self, device) -> 'AttGRUModel':
        """Handle device transfer while preserving special attributes."""
        attrs = {
            'repr_lookup': getattr(self, 'repr_lookup', {}),
            'vocab': getattr(self, 'vocab', None)
        }
        super().to(device)
        for k, v in attrs.items():
            if v is not None:
                setattr(self, k, v)
        return self
