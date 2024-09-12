import logging
import os
import sys

import numpy as np

from CONSTANTS import GET_LOGS_ROOT, GET_PROJECT_ROOT, SESSION

PROJECT_ROOT = GET_PROJECT_ROOT()
LOG_ROOT = GET_LOGS_ROOT()
# Dispose Loggers.
StaticLogger = logging.getLogger("StatisticsRepresentation.")
StaticLogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "StaticLogger.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

StaticLogger.addHandler(console_handler)
StaticLogger.addHandler(file_handler)
StaticLogger.info(
    f"Construct StatisticsLogger success, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
)


class Sequential_TF:
    def __init__(self, id2embed):
        assert isinstance(id2embed, dict)
        self.vocab_size = len(id2embed)
        self.word_dim = id2embed[1].shape
        self.id2embed = id2embed

    def transform(self, instances):
        reprs = []
        for inst in instances:
            repr = np.zeros(self.word_dim)
            for idx in inst.sequence:
                if idx in self.id2embed.keys():
                    repr += self.id2embed[idx]
            reprs.append(repr)
        return np.asarray(reprs, dtype=np.float64)

    def present(self, instances):
        represents = []
        if isinstance(instances, list):
            represents = self.transform(instances)
        else:
            StaticLogger.error(
                f"Sequential TF encoder only accepts list objects as input, got {type(instances)}"
            )
        return represents
