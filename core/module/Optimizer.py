"""Wrapper for optimization and learning rate scheduling."""

from dataclasses import dataclass
from typing import Iterable, List, Callable, Optional
from torch.optim import Adam, lr_scheduler
from torch.nn.parameter import Parameter


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and scheduler.
    
    Args:
        lr: Initial learning rate
        decay_rate: Learning rate decay factor
        decay_steps: Steps between learning rate decays
        beta1: First momentum coefficient for Adam
        beta2: Second momentum coefficient for Adam
        epsilon: Small constant for numerical stability
    """
    lr: float
    decay_rate: float = 0.75
    decay_steps: int = 1000
    beta1: float = 0.9
    beta2: float = 0.9
    epsilon: float = 1e-12
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 < self.decay_rate < 1:
            raise ValueError("Decay rate must be between 0 and 1")
        if self.decay_steps <= 0:
            raise ValueError("Decay steps must be positive")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be between 0 and 1")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


class Optimizer:
    """Enhanced optimizer with integrated learning rate scheduling.
    
    Combines Adam optimizer with exponential learning rate decay in a
    simplified interface. Handles parameter updates, learning rate
    scheduling, and gradient management.
    """
    
    def __init__(
        self,
        parameters: Iterable[Parameter],
        config: Optional[OptimizerConfig] = None
    ):
        """Initialize optimizer with parameters and optional config.
        
        Args:
            parameters: Model parameters to optimize
            config: Optimizer configuration, uses defaults if None
        """
        self.config = config or OptimizerConfig(lr=0.001)
        
        # Initialize Adam optimizer
        self.optimizer = Adam(
            parameters,
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
        
        # Setup learning rate scheduler
        decay_fn = self._create_decay_function()
        self.scheduler = lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=decay_fn
        )
        
    def _create_decay_function(self) -> Callable[[int], float]:
        """Create learning rate decay function.
        
        Returns:
            Function that computes decay factor for each epoch
        """
        def decay_fn(epoch: int) -> float:
            return self.config.decay_rate ** (epoch // self.config.decay_steps)
        return decay_fn
    
    def step(self) -> None:
        """Perform single optimization step.
        
        Updates parameters, adjusts learning rate, and clears gradients.
        """
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def zero_grad(self) -> None:
        """Reset gradients of all optimized parameters."""
        self.optimizer.zero_grad()
    
    @property
    def learning_rates(self) -> List[float]:
        """Current learning rates for all parameter groups.
        
        Returns:
            List of current learning rates
        """
        return self.scheduler.get_last_lr()
    
    @property
    def current_lr(self) -> float:
        """Current learning rate for first parameter group.
        
        Returns:
            Current learning rate
        """
        return self.learning_rates[0]
