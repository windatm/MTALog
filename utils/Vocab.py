import logging
import os
import sys

import numpy as np

from CONSTANTS import LOG_ROOT, SESSION

logger = logging.getLogger("Vocab")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "logger.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info(
    f"Construct logger for Vocab succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
)


class Vocab(object):
    """
    A vocabulary manager that handles token-to-ID mapping, tag encoding, and embedding construction.

    This class supports:
        - Preloading pretrained embeddings from file or dictionary.
        - Token and tag indexing.
        - Handling of special tokens: <pad>, <bos>, <eos>, <oov>.
        - Construction of an embedding matrix with averaged OOV vector.

    Attributes:
        PAD (int): Padding index (always 0).
        START (int): Start-of-sequence token index.
        END (int): End-of-sequence token index.
        UNK (int): Unknown token (OOV) index.
        embeddings (np.ndarray): Embedding matrix aligned with word IDs.
        _word2id (dict): Mapping from word string to integer ID.
        _id2word (list): Reverse mapping from ID to word.
        _tag2id (dict): Mapping from tag label to ID.
        _id2tag (list): Reverse mapping from ID to tag label.
        _embed_dim (int): Embedding vector dimensionality.
    """
    # please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self):
        """
        Initialize the vocabulary with fixed tag mappings ("Normal", "Anomalous").
        Special tokens (<pad>, <bos>, <eos>, <oov>) are also reserved.
        """
        self._id2tag = []
        self._id2tag.append("Normal")
        self._id2tag.append("Anomalous")

        def reverse(x):
            return dict(zip(x, range(len(x))))

        self._tag2id = reverse(self._id2tag)
        # if len(self._tag2id) != len(self._id2tag):
        #     logger.info("serious bug: output tags dumplicated, please check!")
        # logger.info(f"Vocab info: #output tags {self.tag_size}")
        self._embed_dim = 0
        self.embeddings = None

    def load_from_dict(self, id2embed):
        """
        Load embeddings from a dictionary mapping word tokens to embedding vectors.

        Args:
            id2embed (dict): Mapping from word → embedding (np.ndarray).

        Notes:
            - Special tokens are inserted at the beginning.
            - Averages all seen embeddings to initialize the UNK (OOV) vector.
        """
        self._id2word = []
        all_words = set()
        for special_word in ["<pad>", "<bos>", "<eos>", "<oov>"]:
            if special_word not in all_words:
                all_words.add(special_word)
                self._id2word.append(special_word)
        for word, embed in id2embed.items():
            self._embed_dim = embed.shape[0]
            all_words.add(word)
            self._id2word.append(word)

        word_num = len(self._id2word)
        logger.info(f"Total words: {word_num}")
        logger.info(f"The dim of pretrained embeddings: {self._embed_dim}")

        def reverse(x):
            return dict(zip(x, range(len(x))))

        self._word2id = reverse(self._id2word)

        oov_id = self._word2id.get("<oov>")
        if self.UNK != oov_id:
            logger.info("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, self._embed_dim))
        tem_count = 0
        for word, embed in id2embed.items():
            index = self._word2id.get(word)
            vector = np.array(embed, dtype=np.float64)
            embeddings[index] = vector
            embeddings[self.UNK] += vector
            tem_count += 1
        if tem_count != word_num - 4:
            logger.info("Goes wrong when calculating UNK emb!")
        embeddings[self.UNK] = embeddings[self.UNK] / word_num
        self.embeddings = embeddings

    def load_pretrained_embs(self, embfile):
        """
        Load embeddings from a file formatted as:
            <vocab_size> <embedding_dim>
            word1 val1 val2 ...
            word2 val1 val2 ...

        Args:
            embfile (str): Path to the pretrained embedding file.

        Notes:
            - Inserts special tokens before reading from file.
            - Computes average UNK embedding from all vectors.
        """
        embedding_dim = -1
        self._id2word = []
        allwords = set()
        for special_word in ["<pad>", "<bos>", "<eos>", "<oov>"]:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2word.append(special_word)

        with open(embfile, encoding="utf-8") as f:
            line = f.readline()
            vocabSize, embedding_dim = line.strip().split()
            embedding_dim = int(embedding_dim)
            for line in f.readlines():
                values = line.strip().split()
                if len(values) == embedding_dim + 1:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2word.append(curword)
        word_num = len(self._id2word)
        logger.info(f"Total words: {word_num}")
        logger.info(f"The dim of pretrained embeddings: {embedding_dim}")

        def reverse(x):
            return dict(zip(x, range(len(x))))

        self._word2id = reverse(self._id2word)

        if len(self._word2id) != len(self._id2word):
            logger.info("serious bug: words dumplicated, please check!")

        oov_id = self._word2id.get("<oov>")
        if self.UNK != oov_id:
            logger.info("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding="utf-8") as f:
            tem_count = 0
            for line in f.readlines():
                values = line.split()
                if len(values) == embedding_dim + 1:
                    index = self._word2id.get(values[0])
                    vector = np.array(values[1:], dtype="float64")
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
                    tem_count += 1
        if tem_count != word_num - 4:
            logger.info("Goes wrong when calculating UNK emb!")
        embeddings[self.UNK] = embeddings[self.UNK] / word_num

    def word2id(self, xs):
        """
        Convert word(s) to ID(s). Returns UNK index if word not found.

        Args:
            xs (str or List[str]): Word or list of words.

        Returns:
            int or List[int]: Corresponding ID(s).
        """
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        """
        Convert ID(s) back to word(s).

        Args:
            xs (int or List[int]): ID or list of IDs.

        Returns:
            str or List[str]: Corresponding word(s).
        """
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def tag2id(self, xs):
        """
        Convert tag label(s) to ID(s). Supports "Normal", "Anomalous".

        Args:
            xs (str or List[str]): Tag or list of tags.

        Returns:
            int or List[int]: Corresponding ID(s).
        """
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        """
        Convert tag ID(s) back to label(s).

        Args:
            xs (int or List[int]): Tag ID(s).

        Returns:
            str or List[str]: Corresponding tag(s).
        """
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        """
        Return the total number of unique words in the vocabulary.

        Returns:
            int: Size of vocabulary.
        """
        return len(self._id2word)

    @property
    def tag_size(self):
        """
        Return the number of unique output tags (e.g., 2 for binary classification).

        Returns:
            int: Number of tag classes.
        """
        return len(self._id2tag)

    @property
    def word_dim(self):
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            int: Embedding size.
        """
        return self._embed_dim
