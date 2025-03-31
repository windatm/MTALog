from random import random

import numpy as np


def cut_all(instances):
    np.random.shuffle(instances)
    return instances, [], []


def cut_by(train, dev, anomalous_rate=1):
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
