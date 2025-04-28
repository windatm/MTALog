"""Common neural network modules and utilities."""

from dataclasses import dataclass
from typing import List, Optional, Callable, Iterator, Tuple, Any, Union
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from entities.TensorInstances import TInstWithLogits


logger = logging.getLogger(__name__)


def create_batches(
    data: List[Any],
    batch_size: int,
    shuffle: bool = True
) -> Iterator[List[Any]]:
    """Create batches from data list with optional shuffling.
    
    Args:
        data: List of instances to batch
        batch_size: Number of instances per batch
        shuffle: Whether to shuffle data before batching
        
    Yields:
        Batch of instances
    """
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(data), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield [data[i] for i in batch_indices]


@dataclass
class BatchProcessingError(Exception):
    """Custom error for batch processing failures."""
    message: str
    instance_id: Optional[str] = None
    index: Optional[int] = None

    def __str__(self) -> str:
        base = f"Batch processing error: {self.message}"
        if self.instance_id:
            base += f" (instance ID: {self.instance_id})"
        if self.index is not None:
            base += f" at index {self.index}"
        return base


def process_batch(
    instances: List[Any],
    vocab,
    max_seq_len: int = 500
) -> Tuple[Optional[TInstWithLogits], Optional[Tuple[Tensor, ...]]]:
    """Process batch of instances into tensor format.
    
    Args:
        instances: List of instances to process
        vocab: Vocabulary for token/label conversion
        max_seq_len: Maximum sequence length to process
        
    Returns:
        Tuple of (tensor_instance, model_inputs) or (None, None) on error
        
    Raises:
        BatchProcessingError: If batch processing fails
    """
    if not instances:
        raise BatchProcessingError("Empty batch provided")

    try:
        # Validate instances
        for idx, inst in enumerate(instances):
            if not hasattr(inst, 'sequence') or not inst.sequence:
                raise BatchProcessingError(
                    "Missing or empty sequence",
                    getattr(inst, 'id', None),
                    idx
                )
            if not hasattr(inst, 'label'):
                raise BatchProcessingError(
                    "Missing label",
                    getattr(inst, 'id', None),
                    idx
                )

        # Initialize tensor instance
        batch_size = len(instances)
        seq_len = min(max(len(inst.sequence) for inst in instances), max_seq_len)
        tinst = TInstWithLogits(batch_size, seq_len, 2)

        # Process instances
        for idx, inst in enumerate(instances):
            try:
                _process_single_instance(inst, idx, tinst, vocab, max_seq_len)
            except Exception as e:
                logger.warning(
                    f"Error processing instance {getattr(inst, 'id', 'unknown')}: {e}. "
                    "Using default values."
                )
                _set_default_values(tinst, idx, vocab.UNK)

        # Create and validate inputs
        inputs = (tinst.src_words, tinst.src_masks, tinst.word_len)
        if not _validate_inputs(tinst, inputs):
            raise BatchProcessingError("Failed to create valid inputs")

        return tinst, inputs

    except BatchProcessingError:
        raise
    except Exception as e:
        raise BatchProcessingError(f"Unexpected error: {str(e)}")


def _process_single_instance(
    inst: Any,
    idx: int,
    tinst: TInstWithLogits,
    vocab,
    max_seq_len: int
) -> None:
    """Process a single instance into tensor format."""
    # Basic info
    tinst.src_ids.append(str(inst.id))
    
    # Handle label and confidence
    label = getattr(inst, 'predicted', inst.label)
    confidence = getattr(inst, 'confidence', 0.5) * 0.5
    tag_id = vocab.tag2id(label) or 0
    
    tinst.tags[idx, tag_id] = 1 - confidence
    tinst.tags[idx, 1 - tag_id] = confidence
    tinst.g_truth[idx] = tag_id
    
    # Process sequence
    seq_len = len(inst.sequence)
    tinst.word_len[idx] = seq_len
    
    for pos in range(min(seq_len, max_seq_len)):
        token_id = vocab.word2id(inst.sequence[pos])
        tinst.src_words[idx, pos] = token_id
        tinst.src_masks[idx, pos] = 1


def _set_default_values(
    tinst: TInstWithLogits,
    idx: int,
    unk_token: int
) -> None:
    """Set default values for failed instance processing."""
    tinst.word_len[idx] = 1
    tinst.src_words[idx, 0] = unk_token
    tinst.src_masks[idx, 0] = 1
    tinst.tags[idx] = torch.tensor([0.5, 0.5])
    tinst.g_truth[idx] = 0


def _validate_inputs(
    tinst: TInstWithLogits,
    inputs: Tuple[Tensor, ...]
) -> bool:
    """Validate tensor inputs."""
    if not inputs or len(inputs) < 3:
        return False
        
    if not hasattr(tinst, 'inputs'):
        setattr(tinst, '_inputs', inputs)
        
    test_inputs = getattr(tinst, 'inputs', None)
    return test_inputs is not None and len(test_inputs) >= 3


class NonLinear(nn.Module):
    """Linear layer with optional activation.
    
    Args:
        input_size: Input dimension
        hidden_size: Output dimension
        activation: Optional activation function
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: Optional[Callable] = None
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = activation if activation else nn.Identity()
        
        # Initialize weights
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [*, input_size]
            
        Returns:
            Output tensor [*, hidden_size]
        """
        return self.activation(self.linear(x))
