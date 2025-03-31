import logging
import math
import os
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

from CONSTANTS import LOG_ROOT, PROJECT_ROOT, SESSION

logger = logging.getLogger("Statistics_Template_Encoder")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Statistics_Template.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info(
    f"Construct logger for Statistics Template Encoder succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
)


class Template_TF_IDF_without_clean:
    """
    Computes semantic vector representations of log templates using a combination of:
        - Pretrained word embeddings (e.g., Word2Vec),
        - Term Frequency-Inverse Document Frequency (TF-IDF) weighting.

    This encoder treats each template as a small "document" and creates a weighted sum of
    word embeddings based on their importance within the corpus of templates.

    Attributes:
        word2vec_file (str): File path to pretrained word embeddings.
        _word2vec (dict): Dictionary mapping tokens to their embedding vectors.
        vocab_size (int): Dimensionality of the word embeddings.
        total_words (set): Tokens encountered in current run.
        num_oov (set): Out-of-vocabulary tokens in current run.
        total_words_all (set): All tokens encountered historically.
        num_oov_all (set): All OOV tokens encountered historically.
    """
    def __init__(self, word2vec_file):
        """
        Initialize the encoder by loading pretrained word vectors.

        Args:
            word2vec_file (str): Relative path to the word embedding file within the dataset folder.
        """
        self.total_words_all = set()
        self.num_oov_all = set()
        self.total_words = set()
        self.num_oov = set()
        self.word2vec_file = word2vec_file
        self._word2vec = {}
        self.vocab_size = 0
        logger.info(f"Loading word2vec dict from {self.word2vec_file}.")
        self._load_word2vec()

    def transform(self, words):
        """
        Retrieve the vector(s) for a word or list of words from the embedding dictionary.

        Args:
            words (str or List[str]): A single token or a list of tokens.

        Returns:
            np.ndarray or List[np.ndarray]: Corresponding embedding(s), or zero vector(s) if OOV.
        """
        if isinstance(words, list):
            return_list = []
            for word in words:
                self.total_words.add(word)
                self.total_words_all.add(word)
                if word in self._word2vec.keys():
                    return_list.append(self._word2vec[word])
                elif words != "[*]":
                    if words not in self.num_oov:
                        logger.warning(f"OOV: {word}")
                        self.num_oov.add(word)
                        self.num_oov_all.add(word)
                    return_list.append([np.zeros(self.vocab_size)])
            return return_list
        else:
            self.total_words.add(words)
            self.total_words_all.add(words)
            if words in self._word2vec.keys():
                return self._word2vec[words]
            else:
                if words != "[*]":
                    if words not in self.num_oov:
                        logger.warning(f"OOV: {words}")
                        self.num_oov.add(words)
                        self.num_oov_all.add(words)
                return np.zeros(self.vocab_size)

    def _load_word2vec(self):
        """
        Load pretrained word embeddings from file into a dictionary.

        Expected file format:
            Each line contains a word followed by its embedding values (space-separated).

        Logs:
            - Number of embeddings loaded.
            - Embedding dimension.
            - Errors if the file is missing or malformed.
        """
        logger.info("Loading word2vec dict.")
        embed_file = os.path.join(PROJECT_ROOT, f"datasets/{self.word2vec_file}")
        if os.path.exists(embed_file):
            with open(embed_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader.readlines()):
                    try:
                        tokens = line.strip().split()
                        word = tokens[0]
                        embed = np.asarray(tokens[1:], dtype=float)
                        self._word2vec[word] = embed
                        self.vocab_size = len(tokens) - 1
                    except Exception:
                        continue
            pass
        else:
            logger.error(
                f"No pre-trained embedding file({embed_file}) found. Please check."
            )
            sys.exit(2)

    def present(self, id2templates):
        """
        Generate TF-IDF-weighted embeddings for each log template.

        Args:
            id2templates (dict): Mapping from template ID to its raw template string.

        Returns:
            dict: Mapping from template ID to its TF-IDF-weighted embedding vector.
        """
        templates = []
        ids = []
        all_tokens = set()

        self.total_words = set()
        self.num_oov = set()

        for id, template in id2templates.items():
            templates.append(template)
            ids.append(id)
            all_tokens = all_tokens.union(template.split())

        # Calculate IDF score.
        total_templates = len(templates)
        assert total_templates == len(ids)
        token2idf = {}
        for token in all_tokens:
            num_include = 0
            for template in templates:
                if token in template:
                    num_include += 1
            idf = math.log(total_templates / (num_include + 1))
            token2idf[token] = idf

        id2embed = {}
        for id, template in id2templates.items():
            template_tokens = template.split()
            N = len(template_tokens)
            token_counter = Counter(template_tokens)
            template_emb = np.zeros(self.vocab_size)
            if N == 0:
                id2embed[id] = template_emb
                continue
            for token in template_tokens:
                tf = token_counter[token] / N
                idf = token2idf[token]
                embed = self.transform(token)
                template_emb += tf * idf * embed
            id2embed[id] = template_emb

        logger.info(f"Total = {len(self.total_words)}, OOV = {len(self.num_oov)}")
        logger.info(f"OOV Rate = {(len(self.num_oov) / len(self.total_words))}")
        return id2embed
