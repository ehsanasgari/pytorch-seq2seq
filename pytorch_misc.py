"""
Miscellaneous functions that might be useful for pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
import torch
import numpy as np

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
    mask = Variable(torch.zeros(len(context_lens), max(context_lens)))
    if torch.cuda.is_available():
        mask = mask.cuda()

    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1
    return mask


def seq_lengths_from_pad(x, pad_idx):
    lengths = x.size(0) - (x == pad_idx).int().sum(0)[0]
    return lengths.data.tolist()


class PackedShuffledSequence(object):
    """ For sequences that are not sorted """
    def __init__(self, data, seq_lens):
        """
        Initializes a PackedShuffledSequence
        :param data: An array where the sequences are concatenated, ie, data[:t_1] is the first
                     sequence
        :param seq_lens: Lengths of the sequences
        """
        self.data = data
        self.batch_sizes = seq_lens

        self.sorted_lens, fwd_indices = torch.sort(
            torch.IntTensor(seq_lens), dim=0, descending=True,
        )
        self.perm = torch.sort(fwd_indices)[1]
        if torch.cuda.is_available():
            self.perm = self.perm.cuda()

    @classmethod
    def from_padded_seq(cls, x, lengths=None, pad_idx=None):
        """
        Produces a PackedShuffledSequence from an already padded array
        :param x: [max_T, batch_size] array
        :param lengths: seq lengths of the batches
        :param pad_idx: pad index
        :return: desired PackedShuffledSequence
        """
        if lengths is None:
            if pad_idx is None:
                raise ValueError('Must supply some way of getting lengths')
            lengths = seq_lengths_from_pad(x, pad_idx)

        data = x.data.new(sum(lengths), *x.size()[2:]).zero_()
        data = Variable(data)

        data_offset = 0
        for i, seq_l in enumerate(lengths):
            data[data_offset:data_offset+seq_l] = x[:seq_l, i]

        return cls(data, lengths)

    def pad(self):
        batch_size = len(self.sorted_lens)
        max_t = self.sorted_lens[0]

        output = self.data.data.new(max_t, batch_size, *self.data.size()[1:]).zero_()
        output = Variable(output)

        data_offset = 0
        for seq_l, sorted_seq_id in zip(self.batch_sizes, self.perm):
            output[:seq_l, sorted_seq_id] = self.data[data_offset:data_offset + seq_l]
        return output


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start+batch_size, len_l))

def batch_map(f, a, batch_size):
    """
    Maps f over the array a in chunks of batch_size.
    :param f: function to be applied. Must take in a block of
            (batch_size, dim_a) and map it to (batch_size, something).
    :param a: Array to be applied over of shape (num_rows, dim_a).
    :param batch_size: size of each array
    :return: Array of size (num_rows, something).
    """
    rez = []
    for s, e in batch_index_iterator(a.size(0), batch_size, skip_end=False):
        print("Calling on {}".format(a[s:e].size()))
        rez.append(f(a[s:e]))

    # rez = [f(a[s:e]) for s, e in batch_index_iterator(a.size(0),
    #                                                batch_size,
    #                                                skip_end=False)]
    return torch.cat(rez)


def const_row(fill, l):
    input_tok = Variable(torch.LongTensor([fill] * l))
    if torch.cuda.is_available():
        input_tok = input_tok.cuda()
    return input_tok