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
    """
    Sequential Term Frequency (TF) representation of log event sequences using additive embeddings.

    This encoder represents each sequence as the **sum of embedding vectors** of the events
    (tokens/templates) it contains. It assumes that event order is not crucial and focuses on
    content frequency.

    Attributes:
        id2embed (dict): A dictionary mapping event/template IDs to embedding vectors (numpy arrays).
        vocab_size (int): Total number of unique events/templates.
        word_dim (tuple): Dimensionality of each embedding vector.

    Example:
        >>> encoder = Sequential_TF(id2embed)
        >>> X = encoder.present(instances)
    """
    def __init__(self, id2embed):
        """
        Initialize the encoder with a dictionary of event/template embeddings.

        Args:
            id2embed (dict): A mapping from event IDs (int) to embedding vectors (np.ndarray).

        Raises:
            AssertionError: If id2embed is not a dictionary.
        """
        assert isinstance(id2embed, dict)
        self.vocab_size = len(id2embed)
        self.word_dim = id2embed[1].shape
        self.id2embed = id2embed

    def transform(self, instances):
        """
        Convert a list of Instance objects into their embedding representations
        by summing the embeddings of all events in each instance's sequence.

        Args:
            instances (List[Instance]): A list of log instances, each with a `.sequence` attribute.

        Returns:
            np.ndarray: A 2D array of shape [num_instances, embedding_dim], where each row is a sequence embedding.
        """
        reprs = []
        for inst in instances:
            repr = np.zeros(self.word_dim)
            for idx in inst.sequence:
                if idx in self.id2embed.keys():
                    repr += self.id2embed[idx]
            reprs.append(repr)
        return np.asarray(reprs, dtype=np.float64)

    def present(self, instances):
        """
        Public interface to obtain TF-based sequence representations.

        Args:
            instances (List[Instance]): Input instances.

        Returns:
            np.ndarray: Array of sequence-level embeddings.
        """
        represents = []
        if isinstance(instances, list):
            represents = self.transform(instances)
        else:
            logger.error(
                f"Sequential TF encoder only accepts list objects as input, got {type(instances)}"
            )
        return represents
