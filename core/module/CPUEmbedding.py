"""Device-agnostic embedding layer that maintains weights on CPU."""

from dataclasses import dataclass
from typing import Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device as Device


@dataclass
class EmbeddingConfig:
    """Configuration for CPUEmbedding layer.
    
    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Size of each embedding vector
        padding_idx: Optional index for padding token
        std: Standard deviation for weight initialization
    """
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int] = None
    std: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.padding_idx is not None:
            if not -self.num_embeddings <= self.padding_idx < self.num_embeddings:
                raise ValueError("padding_idx must be within ±num_embeddings")


class CPUEmbedding(nn.Module):
    """Embedding layer that keeps weights on CPU while supporting any input device.
    
    This layer is designed for large embedding tables that don't fit in GPU memory.
    The weights stay on CPU, with only the current batch being moved to the target
    device during the forward pass.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding layer.
        
        Args:
            config: Configuration object containing embedding parameters
        """
        super().__init__()
        
        self.config = config
        self._normalize_padding_idx()
        
        # Initialize weights on CPU
        self.weight = nn.Parameter(
            torch.empty(config.num_embeddings, config.embedding_dim, 
                       dtype=torch.float32, device='cpu')
        )
        self.reset_parameters()

    def _normalize_padding_idx(self) -> None:
        """Convert negative padding index to positive."""
        if self.config.padding_idx is not None and self.config.padding_idx < 0:
            self.config.padding_idx += self.config.num_embeddings

    def reset_parameters(self) -> None:
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.weight, mean=0, std=self.config.std)
        if self.config.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.config.padding_idx].fill_(0)

    def _apply(self, fn: Any) -> 'CPUEmbedding':
        """Prevent weight movement to GPU while allowing other parameters to move.
        
        Args:
            fn: Function to apply to parameters
            
        Returns:
            Self for chaining
        """
        # Skip CUDA transfers
        if 'cuda' in str(fn):
            return self
            
        # Apply to children and non-weight parameters
        for module in self.children():
            module._apply(fn)
            
        for name, param in self.named_parameters():
            if param is not None and name != 'weight':
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
                    
        # Apply to buffers
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
                
        return self

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Compute embeddings for input indices.
        
        The weights are temporarily moved to the input device for computation
        and the result is returned on the same device.
        
        Args:
            input_tensor: Integer tensor of token indices
            
        Returns:
            Tensor of embeddings with shape [..., embedding_dim]
        """
        weight = self.weight.to(input_tensor.device, non_blocking=True)
        return F.embedding(input_tensor, weight, self.config.padding_idx)

    def to(self, *args, **kwargs) -> 'CPUEmbedding':
        """Override device movement to keep weights on CPU."""
        return self

    def cuda(self, device: Optional[Union[int, Device]] = None) -> 'CPUEmbedding':
        """Override cuda() to keep weights on CPU."""
        return self

    def cpu(self) -> 'CPUEmbedding':
        """Override cpu() to maintain consistency."""
        return self

    def extra_repr(self) -> str:
        """Return string representation of layer configuration."""
        return (f"num_embeddings={self.config.num_embeddings}, "
                f"embedding_dim={self.config.embedding_dim}, "
                f"padding_idx={self.config.padding_idx}")
