import torch
from torch.autograd import Variable


class TInstWithLogits:
    def __init__(self, batch_size, slen, tag_size):
        self.src_ids = []
        self.src_words = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.src_masks = Variable(
            torch.Tensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.tags = Variable(
            torch.FloatTensor(batch_size, tag_size).zero_(), requires_grad=False
        )
        self.g_truth = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        # Explicitly store the inputs tuple to avoid issues with the property
        self._inputs_cache = None

    def to(self, device):
        """Move all tensors to the specified device"""
        self.src_words = self.src_words.to(device)
        self.src_masks = self.src_masks.to(device)
        self.tags = self.tags.to(device)
        self.g_truth = self.g_truth.to(device)
        self.word_len = self.word_len.to(device)
        # Clear the inputs cache since tensors have moved
        self._inputs_cache = None
        return self

    @property
    def inputs(self):
        """Return a tuple of tensors needed for model input
        
        Returns:
            tuple: (src_words, src_masks, word_len)
        """
        # Use cached value if available
        if self._inputs_cache is not None:
            return self._inputs_cache
            
        # Create new tuple and cache it
        inputs_tuple = (self.src_words, self.src_masks, self.word_len)
        self._inputs_cache = inputs_tuple
        return inputs_tuple

    @inputs.setter
    def inputs(self, value):
        """Set the inputs tuple directly
        
        Args:
            value (tuple): A tuple of (src_words, src_masks, word_len)
        """
        if not isinstance(value, tuple) or len(value) < 3:
            raise ValueError("inputs must be a tuple with at least 3 elements")
            
        self.src_words, self.src_masks, self.word_len = value[:3]
        self._inputs_cache = value

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.tags

    @property
    def truth(self):
        return self.g_truth


class TInstWithoutLogits:
    def __init__(self, batch_size, slen, tag_size):
        self.src_words = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.src_masks = Variable(
            torch.Tensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.tags = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )

    def to(self, device):
        self.src_words = self.src_words.to(device)
        self.src_masks = self.src_masks.to(device)
        self.tags = self.tags.to(device)
        self.word_len = self.word_len.to(device)


    @property
    def inputs(self):
        return self.src_words, self.src_masks, self.word_len

    @property
    def targets(self):
        return self.tags

    @property
    def truth(self):
        return self.tags
