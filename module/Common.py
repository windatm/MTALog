import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from entities.TensorInstances import TInstWithLogits


def batch_slice(data, batch_size):
    """
    Slice the input data into smaller batches of specified size.

    Args:
        data (list): A list of data instances (e.g., log sequences).
        batch_size (int): Number of instances per batch.

    Yields:
        list: A batch containing up to 'batch_size' instances.
    """
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield insts


def insts_numberize(insts, vocab):
    """
    Convert a list of data instances into numerical (ID-based) form using a vocabulary.

    Args:
        insts (list): List of raw data instances.
        vocab (Vocabulary): A vocabulary object with word2id and tag2id mappings.

    Yields:
        tuple: A tuple (srcids, tagid, inst) representing word IDs, label ID, and the original instance.
    """
    for inst in insts:
        yield inst2id(inst, vocab)


def inst2id(inst, vocab):
    """
    Convert a single instance into numeric ID representation.

    Args:
        inst (Instance): A data instance containing source events and label.
        vocab (Vocabulary): Vocabulary object mapping tokens and tags to IDs.

    Returns:
        tuple: (srcids, tagid, inst) – source token IDs, tag ID, and original instance.
    """
    srcids = vocab.word2id(inst.src_events)
    tagid = vocab.tag2id(inst.tag)
    return srcids, tagid, inst


def data_iter(data, batch_size, shuffle=True):
    """
    Create an iterator over data batches with optional shuffling.

    Args:
        data (list): List of input instances.
        batch_size (int): Number of instances per batch.
        shuffle (bool): Whether to shuffle the data and batches.

    Yields:
        list: A batch of instances.
    """
    batched_data = []
    if shuffle:
        np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))
    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def generate_tinsts_binary_label(batch_insts, vocab, if_evaluate=False):
    """
    Generate tensor-based batch data for binary classification tasks.

    Args:
        batch_insts (list): List of input instances.
        vocab (Vocabulary): Vocabulary object with word2id and tag2id mappings.
        if_evaluate (bool): Flag for evaluation mode (not used internally).

    Returns:
        TInstWithLogits: A structured tensor containing inputs, masks, lengths, and label targets.
    """

    slen = len(batch_insts[0].sequence)
    batch_size = len(batch_insts)
    for b in range(1, batch_size):
        cur_slen = len(batch_insts[b].sequence)
        if cur_slen > slen:
            slen = cur_slen
    tinst = TInstWithLogits(batch_size, slen, 2)
    b = 0
    for inst in batch_insts:
        tinst.src_ids.append(str(inst.id))
        confidence = 0.5 * inst.confidence
        if inst.predicted == "":
            inst.predicted = inst.label
        tinst.tags[b, vocab.tag2id(inst.predicted)] = 1 - confidence
        tinst.tags[b, 1 - vocab.tag2id(inst.predicted)] = confidence
        tinst.g_truth[b] = vocab.tag2id(inst.predicted)
        cur_slen = len(inst.sequence)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            if index >= 500:
                break
            tinst.src_words[b, index] = vocab.word2id(inst.sequence[index])
            tinst.src_masks[b, index] = 1
        b += 1
    return tinst


def batch_variable_inst(insts, tagids, tag_logits, id2tag):
    """
    Match model predictions with ground truth labels for evaluation.

    Args:
        insts (list): List of original instances.
        tagids (list): Predicted tag IDs.
        tag_logits (tensor): Prediction scores.
        id2tag (dict): Mapping from tag IDs back to tag strings.

    Yields:
        tuple: (instance, prediction_correct) – a boolean flag indicating correctness.
    """
    if tag_logits is None:
        print("No prediction made, please check.")
        sys.exit(-1)
    for inst, tagid, tag_logit in zip(insts, tagids, tag_logits):
        pred_label = id2tag[tagid]
        yield inst, pred_label == inst.label


def tensor_2_np(t):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Tensor converted to NumPy format.
    """
    return t.detach().cpu().numpy()


def orthonormal_initializer(output_size, input_size):
    """
    Generate an orthonormal matrix for weight initialization using iterative optimization.

    This function is adapted from Timothy Dozat's parser implementation.

    Args:
        output_size (int): Number of output units.
        input_size (int): Number of input features.

    Returns:
        np.ndarray: A transposed orthonormal matrix of shape [output_size, input_size].
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = 0.1
    eps = 0.05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= (
                lr
                * Q.dot(QTQmI)
                / (
                    np.abs(
                        Q2
                        + Q2.sum(axis=0, keepdims=True)
                        + Q2.sum(axis=1, keepdims=True)
                        - 1
                    )
                    + eps
                )
            )
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print("Orthogonal pretrainer loss: %.2e" % loss)
    else:
        print("Orthogonal pretrainer failed, using non-orthogonal random matrix")
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def drop_input_independent(word_embeddings, dropout_emb):
    """
    Apply independent dropout noise to word embeddings.

    Args:
        word_embeddings (Tensor): Tensor of shape [batch_size, seq_len, emb_dim].
        dropout_emb (float): Dropout probability.

    Returns:
        Tensor: Word embeddings with dropout applied independently per element.
    """
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    """
    Apply dropout with a shared mask across all time steps in a sequence.

    Args:
        inputs (Tensor): Input tensor of shape [seq_len, batch_size, hidden_size] or [batch_size, seq_len, hidden_size].
        dropout (float): Dropout probability.
        batch_first (bool): Whether the input format is batch-first.

    Returns:
        Tensor: Input tensor with shared-mask dropout applied.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = (
        torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    )
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class NonLinear(nn.Module):
    """
    A neural module that applies a linear transformation followed by an optional activation function.

    This module is useful as a lightweight feedforward layer that can be inserted after recurrent,
    attention, or convolutional blocks to project feature representations into a desired space
    (e.g., logits or embedding space).

    Args:
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Dimensionality of the output (projected) features.
        activation (callable, optional): A callable activation function (e.g., torch.relu). 
                                         If None, no activation is applied (i.e., identity function).

    Raises:
        ValueError: If the provided activation is not callable.
    """
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError(
                    "activation must be callable: type={}".format(type(activation))
                )
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        """
        Forward pass through the linear layer followed by activation.

        Args:
            x (Tensor): Input tensor of shape [*, input_size].

        Returns:
            Tensor: Output tensor of shape [*, hidden_size] after linear + activation.
        """
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        """
        Initialize the weights using an orthonormal initializer and set biases to zero.

        Orthonormal initialization improves training stability, especially in recurrent and deep models.
        """

        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))

        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.data.copy_(torch.from_numpy(b))


class Biaffine(nn.Module):
    """
    Implements a biaffine transformation layer between two input tensors.

    This module is commonly used in structured prediction tasks, such as dependency parsing
    or relation extraction, where pairwise interactions between token representations are modeled.

    The computation can be summarized as:
        score(i, j) = x_i^T W y_j + U x_i + V y_j + b
    where W is a 3D tensor, and the other terms are optional biases depending on the `bias` argument.

    Args:
        in1_features (int): Dimensionality of the first input tensor (input1).
        in2_features (int): Dimensionality of the second input tensor (input2).
        out_features (int): Number of output channels (typically number of relation types or scores).
        bias (tuple): A pair of booleans indicating whether to append a bias term (i.e., a constant '1')
                      to input1 and/or input2.

    Example:
        input1: [batch_size, len1, in1_features]
        input2: [batch_size, len2, in2_features]
        output: [batch_size, len2, len1, out_features]
    """

    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(
            in_features=self.linear_input_size,
            out_features=self.linear_output_size,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weight matrix of the internal linear layer with zeros.

        Note:
            A more expressive model could use random or orthogonal initialization instead.
        """
        W = np.zeros(
            (self.linear_output_size, self.linear_input_size), dtype=np.float32
        )
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        """
        Performs a biaffine transformation between two input tensors.

        Args:
            input1 (Tensor): Tensor of shape [batch_size, len1, in1_features]
            input2 (Tensor): Tensor of shape [batch_size, len2, in2_features]

        Returns:
            Tensor: Output of shape [batch_size, len2, len1, out_features]
                    representing biaffine scores between elements in input1 and input2.
        """
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + "in1_features="
            + str(self.in1_features)
            + ", in2_features="
            + str(self.in2_features)
            + ", out_features="
            + str(self.out_features)
            + ")"
        )
