from random import random

import numpy as np


def cut_all(instances):
    """
    Shuffle the entire dataset without splitting.

    This function returns the full dataset as training data,
    and leaves the dev and test sets empty.

    Args:
        instances (list): A list of data instances.

    Returns:
        tuple:
            - list: Shuffled instances (used as training data).
            - list: Empty list (dev set).
            - list: Empty list (test set).
    """
    np.random.shuffle(instances)
    return instances, [], []


def cut_by(train, dev, anomalous_rate=1):
    """
    Returns a customized data splitting function that partitions a dataset into
    training, development, and test sets based on given proportions.

    Additionally, it can **downsample anomalous instances** in the training set 
    by keeping only a fraction (controlled by `anomalous_rate` ∈ [0, 1]).

    Args:
        train (float): Proportion of data to be used for training (0 < train < 1).
        dev (float): Proportion of data to be used for development (0 < dev < 1).
        anomalous_rate (float): Probability of keeping an anomalous instance in the training set.

    Returns:
        function: A function `cut(instances)` that applies the defined split and filtering.

    Example:
        splitter = cut_by(0.7, 0.1, anomalous_rate=0.2)
        train, dev, test = splitter(instances)
    """
    def cut(instances):
        nonlocal train, dev, anomalous_rate
        dev_split = int(dev * len(instances))
        train_split = int(train * len(instances))
        train_dev = instances[: (train_split + dev_split)]
        np.random.shuffle(train_dev)
        dev = train_dev[train_split:]
        train = train_dev[:train_split]
        test = instances[(train_split + dev_split) :]
        temp = []
        for ins in train:
            if ins.label == "Anomalous":
                ran = random()
                if ran > anomalous_rate:
                    continue
            temp.append(ins)
        train = temp
        return train, dev, test

    return cut
