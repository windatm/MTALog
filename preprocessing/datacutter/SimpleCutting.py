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
        tuple: (support_set, query_set)
    """
    normal_blocks = [ins for ins in instances if ins.label == "Normal"]
    k = int(normal_ratio * len(normal_blocks))

    np.random.shuffle(normal_blocks)
    support_set = normal_blocks[:k]
    support_ids = {ins.id for ins in support_set}
    query_set = [ins for ins in instances if ins.id not in support_ids]

    return support_set, query_set


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
        raise ValueError("Empty query set provided")
        
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("Sample ratio must be between 0 and 1")
        
    sample_size = int(len(query_set) * sample_ratio)
    indices = np.random.choice(len(query_set), size=sample_size, replace=False)
    
    sampled_query = [query_set[i] for i in indices]
    logger.info(f"Sampled {len(sampled_query)} instances from query set of size {len(query_set)}")
    
    return sampled_query
