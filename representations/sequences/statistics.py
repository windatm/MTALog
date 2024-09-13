import logging
import os
import sys

import numpy as np

from CONSTANTS import LOG_ROOT, SESSION

logger = logging.getLogger("StatisticsRepresentation.")
logger.setLevel(logging.DEBUG)
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

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info(
    f"Construct logger for Statistics Representation succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
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
            logger.error(
                f"Sequential TF encoder only accepts list objects as input, got {type(instances)}"
            )
        return represents
