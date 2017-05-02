"""
Miscellaneous functions that might be useful for pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
import torch


def packed_seq_iter(packed_seq):
    """
    Returns an iterator for a PackedSequence, where Time is first dim
    :param packed_seq:
    :return: Iterator that goes through the first sequence by time
    """
    data, batch_sizes = packed_seq
    i = 0
    for b in batch_sizes:
        yield data[i:i + b]
        i += b


def transpose_batch_sizes(lengths):
    """
    Given a list of sequence lengths per batch size (ie for an RNN where sequence lengths vary),
     converts this into a list of batch sizes per timestep
    :param lengths: Sorted (descending order) list of ints
    :return: A list of length lengths[0]
    """
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    end_inds = []
    for i in range(max_len):
        while (length_pointer > 0) and lengths[length_pointer] <= i:
            length_pointer -= 1
        end_inds.append(length_pointer + 1)
    return end_inds


def rnn_mask(context_lens):
    """
    Creates a mask for variable length sequences
    """
    mask = Variable(torch.zeros(len(context_lens), context_lens[0]))
    if torch.cuda.is_available():
        mask = mask.cuda()

    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1
    return mask
