import logging
import os
import sys

import torch
import torch.mps
import torch.nn as nn
from torch.nn.parameter import Parameter

from CONSTANTS import DEVICE, LOG_ROOT, SESSION
from module.Attention import LinearAttention
from module.Common import NonLinear, drop_input_independent
from module.CPUEmbedding import CPUEmbedding

class AttGRUModel(nn.Module): 
    """
    A neural network model that combines bidirectional GRU with an attention mechanism for sequence-level classification.

    This architecture is particularly suited for log-based anomaly detection and event classification tasks,
    where temporal dependencies and localized attention over token sequences are critical.

    Components:
        - Pretrained static embeddings (non-trainable).
        - Bidirectional multi-layer GRU encoder.
        - Learnable global attention query vector.
        - Linear attention mechanism over GRU outputs.
        - Final projection layer for classification.

    Args:
        vocab (Vocab): Vocabulary object containing pretrained embeddings and metadata.
        lstm_layers (int): Number of GRU layers.
        lstm_hiddens (int): Number of hidden units per GRU direction.
        dropout (float): Dropout rate applied to input embeddings.
        is_backup (bool): If True, disables verbose logging.
    """

    # Dispose Loggers.
    _logger = logging.getLogger("AttGRU")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )
 
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "AttGRU.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        f"Construct logger for Attention-Based GRU succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
    )

    @property
    def logger(self):
        return AttGRUModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0.0, is_backup=False):
        """
        Initialize the attention-based GRU model.

        Args:
            vocab (Vocab): Vocabulary object containing embedding matrix, size, and dimension.
            lstm_layers (int): Number of GRU layers.
            lstm_hiddens (int): GRU hidden size for each direction (forward & backward).
            dropout (float): Dropout rate for regularization.
            is_backup (bool): Whether to suppress logging (e.g., during evaluation or backup runs).
        """
        super(AttGRUModel, self).__init__()
        self.dropout = dropout
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        # Loads pretrained embeddings from vocab.embeddings
        self.word_embed = CPUEmbedding(
            vocab_size, word_dims, padding_idx=vocab_size - 1
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        # Freezes the embedding weights
        self.word_embed.weight.requires_grad = False

        if not is_backup:
            self.logger.info("==== Model Parameters ====")
            self.logger.info(f"Input Dimension: {word_dims}")
            self.logger.info(f"Hidden Size: {lstm_hiddens}")
            self.logger.info(f"Num Layers: {lstm_layers}")
            self.logger.info(f"Dropout: {round(dropout, 3)}")
        self.rnn = nn.GRU(
            input_size=word_dims,
            hidden_size=lstm_hiddens,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # bidirectional -> forward + backward
        self.sent_dim = 2 * lstm_hiddens
        # attention vector
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        # attention of query (atten_guide) and key/value (hidden vector GRU) -> weight
        self.atten = LinearAttention(
            tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim
        )
        self.proj = NonLinear(self.sent_dim, 2)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        """
        Reset the pretrained embedding layer with a new embedding matrix.

        Args:
            vocab (Vocab): Vocabulary object to access padding index.
            pretrained_embedding (np.ndarray): New pretrained embedding matrix of shape [vocab_size, word_dim].
        """
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(
            vocab.vocab_size, word_dims, padding_idx=vocab.PAD
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        """
        Forward pass of the model. Computes the attention-weighted GRU encoding and final logits.

        Args:
            inputs (tuple): A 3-tuple containing:
                - words (Tensor): Token IDs of shape [batch_size, seq_len].
                - masks (Tensor): Attention masks of shape [batch_size, seq_len].
                - word_len (Tensor): Sequence lengths [batch_size].

        Returns:
            Tensor: Logits of shape [batch_size, 2] representing binary class scores.
        """

        # words: Tensor of token indices [batch_size, seq_len]
        # masks: Boolean tensor [batch_size, seq_len] indicating valid tokens (used in attention)
        words, masks, word_len = inputs
        # Converts token indices into dense vectors using a fixed pretrained embedding
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        if torch.cuda.is_available():
            embed = embed.cuda(DEVICE)
        elif hasattr(torch.mps, "is_available") and torch.mps.is_available():
            embed = embed.to(DEVICE)
        batch_size = embed.size(0)
        # Learnable vector used as a query to compute attention over the GRU outputs.
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        # Processes the embedded sequence using a multi-layer Bi-GRU
        # hiddens: shape [batch_size, seq_len, 2 * hidden_size]
        hiddens, state = self.rnn(embed)
        # Attention weights for each token, shape [batch_size, seq_len, 1]
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        # Multiplies attention weights with GRU outputs → weighted sum.
        # Produces one vector per sequence, shape [batch_size, sent_dim].
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs  # , represents
