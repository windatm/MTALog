import random as rand
import logging
import numpy as np

logger = logging.getLogger(__name__)


def cut_all(instances):
    """
    Shuffle the entire dataset without splitting.

    This function returns the full dataset as training data,
    and leaves the val and test sets empty.

    Args:
        instances (list): A list of data instances.

    Returns:
        tuple:
            - list: Shuffled instances (used as training data).
            - list: Empty list (val set).
            - list: Empty list (test set).
    """
    np.random.shuffle(instances)
    return instances, [], []


def cut_by(train, val, anomalous_rate=1, random_seed=None):
    """
    Returns a customized data splitting function that partitions a dataset into
    training, validation, and test sets based on given proportions.

    Args:
        train (float): Proportion of data to be used for training (0 < train <= 1).
        val (float): Proportion of data to be used for validation (0 <= val < 1).
        anomalous_rate (float): Probability of keeping an anomalous instance in the training set.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        function: A function `cut(instances)` that applies the defined split and filtering.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        rand.seed(random_seed)

    def cut(instances):
        nonlocal train, val, anomalous_rate
        if not instances:
            raise ValueError("Empty instance list provided")
            
        # Validate proportions
        if train <= 0 or train > 1:
            raise ValueError("Train proportion must be between 0 and 1")
        if val < 0 or val >= 1:
            raise ValueError("Validation proportion must be between 0 and 1")
        if train + val > 1:
            raise ValueError("Train + validation proportion must be less than or equal to 1")

        val_split = int(val * len(instances))
        train_split = int(train * len(instances))
        
        # Shuffle instances
        np.random.shuffle(instances)
        
        # Split data
        train_val = instances[: (train_split + val_split)]
        val = train_val[train_split:]
        train = train_val[:train_split]
        test = instances[(train_split + val_split):]

        # Filter anomalous instances in training set
        filtered_train = []
        for ins in train:
            if ins.label == "Anomalous":
                if rand.random() <= anomalous_rate:
                    filtered_train.append(ins)
            else:
                filtered_train.append(ins)

        logger.info(f"Split sizes - Train: {len(filtered_train)}, Val: {len(val)}, Test: {len(test)}")
        logger.info(f"Anomaly rate in training: {sum(1 for x in filtered_train if x.label == 'Anomalous')/max(len(filtered_train), 1):.2%}")
        
        return filtered_train, val, test

    return cut


def fewshot_split(instances, normal_ratio):
    """
    Split few Normal samples for building support set (few-shot).
    Remaining samples: query set.

    Args:
        instances (list[Instance]): all blocks (with label).
        normal_ratio (float): normal sample ratio for support set.

    Returns:
        tuple: (support_set, remaining_normal)
    """
    # Handle both string and numeric labels
    normal_blocks = [ins for ins in instances if getattr(ins, 'label', None) == "Normal" or getattr(ins, 'label', None) == 0]
    
    # Make sure we have at least one sample
    k = max(1, int(normal_ratio * len(normal_blocks)))

    # Shuffle the normal blocks for random selection
    np.random.shuffle(normal_blocks)
    
    # Select k samples for support set (ONLY normal samples)
    support_set = normal_blocks[:k]
    
    # Create a set of IDs from the support set for fast lookup
    support_ids = {getattr(ins, 'id', id(ins)) for ins in support_set}
    
    # Return all normal instances that are not in the support set
    remaining_normal = [ins for ins in normal_blocks if getattr(ins, 'id', id(ins)) not in support_ids]
    
    return support_set, remaining_normal


def create_query_set(remaining_normal, malicious_instances, normal_ratio=0.5, sample_ratio=1.0, random_seed=None):
    """
    Create a query set that includes both normal and malicious data.
    
    Args:
        remaining_normal (list[Instance]): Normal instances not used in support set.
        malicious_instances (list[Instance]): Malicious/anomalous instances.
        normal_ratio (float): Ratio of normal instances to include in query set.
        sample_ratio (float): Overall sampling ratio from total available instances.
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        list[Instance]: Combined query set with both normal and malicious instances.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Sample from normal instances
    normal_sample_size = max(1, int(len(remaining_normal) * normal_ratio * sample_ratio))
    normal_sample_size = min(normal_sample_size, len(remaining_normal))
    
    if normal_sample_size > 0 and remaining_normal:
        normal_indices = np.random.choice(len(remaining_normal), size=normal_sample_size, replace=False)
        sampled_normal = [remaining_normal[i] for i in normal_indices]
    else:
        sampled_normal = []
    
    # Sample from malicious instances
    malicious_sample_size = max(1, int(len(malicious_instances) * sample_ratio))
    malicious_sample_size = min(malicious_sample_size, len(malicious_instances))
    
    if malicious_sample_size > 0 and malicious_instances:
        malicious_indices = np.random.choice(len(malicious_instances), size=malicious_sample_size, replace=False)
        sampled_malicious = [malicious_instances[i] for i in malicious_indices]
    else:
        sampled_malicious = []
    
    # Combine the samples
    query_set = sampled_normal + sampled_malicious
    
    # Shuffle the final query set to mix normal and malicious instances
    np.random.shuffle(query_set)
    
    logger.info(f"Created query set with {len(sampled_normal)} normal and {len(sampled_malicious)} malicious instances")
    
    return query_set


def sample_query_set(query_set, sample_ratio=0.1, random_seed=None):
    """
    Sample a subset of query set for testing.

    Args:
        query_set (list[Instance]): Full query set.
        sample_ratio (float): Ratio of samples to keep (0 < ratio <= 1).
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        list[Instance]: Sampled query set.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if not query_set:
        logger.warning("Empty query set provided to sample_query_set. Returning empty list.")
        return []
        
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("Sample ratio must be between 0 and 1")
        
    sample_size = max(1, int(len(query_set) * sample_ratio))
    sample_size = min(sample_size, len(query_set))
    indices = np.random.choice(len(query_set), size=sample_size, replace=False)
    
    sampled_query = [query_set[i] for i in indices]
    logger.info(f"Sampled {len(sampled_query)} instances from query set of size {len(query_set)}")
    
    return sampled_query


def cut_sequential(train, val):
    """
    Returns a sequential data-splitting function that partitions a dataset into
    training, validation, and test sets without shuffling, based on given proportions.
    
    Useful for time-ordered log data where preserving sequence is important.

    Args:
        train (float): Proportion of data to be used for training (0 < train <= 1).
        val (float): Proportion of data to be used for validation (0 <= val < 1).

    Returns:
        function: A function `cut(instances)` that applies the sequential split.
    """
    def cut(instances):
        nonlocal train, val
        if not instances:
            raise ValueError("Empty instance list provided")
            
        # Validate proportions
        if train <= 0 or train > 1:
            raise ValueError("Train proportion must be between 0 and 1")
        if val < 0 or val >= 1:
            raise ValueError("Validation proportion must be between 0 and 1")
        if train + val > 1:
            raise ValueError("Train + validation proportion must be less than or equal to 1")

        # Calculate split points
        val_split = int(val * len(instances))
        train_split = int(train * len(instances))
        
        # Split data sequentially (without shuffling)
        train_data = instances[:train_split]
        val_data = instances[train_split:train_split + val_split]
        test_data = instances[train_split + val_split:]

        logger.info(f"Sequential split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data

    return cut
