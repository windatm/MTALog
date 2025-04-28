"""Dataset splitting utilities for log analysis."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, TypeVar
import logging
from numpy.random import RandomState
import numpy as np

# Define generic instance type
Instance = TypeVar('Instance')
DataSplit = Tuple[List[Instance], List[Instance], List[Instance]]

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting.
    
    Args:
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        anomaly_ratio: Probability of keeping anomalous instances
        random_seed: Optional seed for reproducibility
        sample_ratio: Proportion of data to sample
    """
    train_ratio: float
    val_ratio: float
    anomaly_ratio: float = 1.0
    random_seed: Optional[int] = None
    sample_ratio: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate split configuration."""
        if not 0 < self.train_ratio <= 1:
            raise ValueError("Train ratio must be between 0 and 1")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("Validation ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio > 1:
            raise ValueError("Total split ratio exceeds 1")
        if not 0 <= self.anomaly_ratio <= 1:
            raise ValueError("Anomaly ratio must be between 0 and 1")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("Sample ratio must be between 0 and 1")


def cut_all(instances: List[Instance]) -> DataSplit:
    """Shuffle the entire dataset without splitting.
    
    Returns the full dataset as training data and empty validation/test sets.
    
    Args:
        instances: List of data instances
        
    Returns:
        Tuple of (shuffled_instances, [], [])
    """
    shuffled = instances.copy()
    np.random.shuffle(shuffled)
    return shuffled, [], []

def create_query_set(
    normal_instances: List[Instance],
    anomalous_instances: List[Instance],
    config: Optional[SplitConfig] = None
) -> List[Instance]:
    """Create balanced query set from normal and anomalous instances.
    
    Args:
        normal_instances: Normal instances
        anomalous_instances: Anomalous instances
        config: Optional configuration for ratios and seed
        
    Returns:
        Combined and shuffled query set
    """
    if not config:
        config = SplitConfig(train_ratio=0.5, val_ratio=0.0)
        
    rng = RandomState(config.random_seed)
    
    # Sample normal instances
    n_normal = max(1, int(len(normal_instances) * config.train_ratio * config.sample_ratio))
    n_normal = min(n_normal, len(normal_instances))
    normal_sample = (
        rng.choice(normal_instances, size=n_normal, replace=False).tolist()
        if normal_instances else []
    )
    
    # Sample anomalous instances
    n_anomalous = max(1, int(len(anomalous_instances) * config.sample_ratio))
    n_anomalous = min(n_anomalous, len(anomalous_instances))
    anomalous_sample = (
        rng.choice(anomalous_instances, size=n_anomalous, replace=False).tolist()
        if anomalous_instances else []
    )
    
    # Combine and shuffle
    query_set = normal_sample + anomalous_sample
    rng.shuffle(query_set)
    
    logger.info(
        f"Created query set with {len(normal_sample)} normal and "
        f"{len(anomalous_sample)} anomalous instances"
    )
    
    return query_set


def sample_query_set(
    query_set: List[Instance],
    sample_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> List[Instance]:
    """Sample a subset of query set for testing.
    
    Args:
        query_set: Full query set
        sample_ratio: Ratio of samples to keep (0 < ratio <= 1)
        random_seed: Optional seed for reproducibility
        
    Returns:
        Sampled query set
    """
    if not query_set:
        logger.warning("Empty query set provided. Returning empty list.")
        return []
        
    if not 0 < sample_ratio <= 1:
        raise ValueError("Sample ratio must be between 0 and 1")
        
    rng = RandomState(random_seed)
    
    sample_size = max(1, int(len(query_set) * sample_ratio))
    sample_size = min(sample_size, len(query_set))
    
    indices = rng.choice(len(query_set), size=sample_size, replace=False)
    sampled = [query_set[i] for i in indices]
    
    logger.info(f"Sampled {len(sampled)} instances from query set of size {len(query_set)}")
    return sampled
