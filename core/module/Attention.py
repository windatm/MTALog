"""Attention mechanisms for neural networks."""

from typing import Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(
    vector: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    mask_fill_value: float = -1e32
) -> torch.Tensor:
    """Apply masked softmax to vector.
    
    Args:
        vector: Input tensor
        mask: Boolean mask (1 for valid, 0 for masked)
        dim: Dimension to apply softmax
        mask_fill_value: Value to fill masked positions
        
    Returns:
        Tensor with masked softmax applied
    """
    if mask is None:
        return F.softmax(vector, dim=dim)
        
    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
        
    masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
    return F.softmax(masked_vector, dim=dim)


class LinearAttention(nn.Module):
    """Linear attention mechanism.
    
    Computes attention between a query vector and key matrix:
    attention = softmax(query @ W @ keys.T)
    
    Args:
        query_dim: Query vector dimension
        key_dim: Key vector dimension
        normalize: Whether to apply softmax normalization
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        normalize: bool = True
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.normalize = normalize
        
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention weights.
        
        Args:
            query: Query tensor [batch_size, query_dim]
            keys: Key tensor [batch_size, seq_len, key_dim]
            mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Attention weights [batch_size, seq_len]
        """
        # Project query to key space
        query_proj = self.query_proj(query)
        
        # Compute attention scores
        scores = torch.bmm(
            query_proj.unsqueeze(1),
            keys.transpose(1, 2)
        ).squeeze(1)
        
        # Apply mask and normalization
        if self.normalize:
            return masked_softmax(scores, mask)
        return scores
