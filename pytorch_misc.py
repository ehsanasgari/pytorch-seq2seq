"""
Miscellaneous functions that might be useful for pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
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
    num_batches = len(context_lens)
    max_batch_size = max(context_lens)

    mask = torch.FloatTensor(num_batches, max_batch_size).zero_()
    if torch.cuda.is_available():
        mask = mask.cuda()
    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1.0
    mask = Variable(mask)
    return mask


def seq_lengths_from_pad(x, pad_idx):
    lengths = x.size(0) - (x == pad_idx).int().sum(0)[0]
    return lengths.data.tolist()


class PackedSortedSequence(object):
    """ For sequences that are not sorted """
    def __init__(self, x, seq_lens=None, pad_idx=None):
        """
        Initializes a PackedShuffledSequence
        :param data: Several options:
                     1. can be a [max_T, batch_size] array that is padded by pad_idx, or with
                        sequence lengths seq_lens
                     2. can be an array where the sequences are concatenated, ie, data[:t_1] is the
                        first sequence
        :param seq_lens: Lengths of the sequences
        :param pad_idx: Pad index
        """
        if seq_lens is None:
            if pad_idx is None:
                raise ValueError('Must supply some way of getting lengths')
            seq_lens = seq_lengths_from_pad(x, pad_idx)

        self.seq_lens = seq_lens

        self.sorted_lens, fwd_indices = torch.sort(
            torch.IntTensor(seq_lens), dim=0, descending=True,
        )
        self.perm = torch.sort(fwd_indices)[1]
        if torch.cuda.is_available():
            self.perm = self.perm.cuda()

        use_concat = x.size(0) == sum(self.seq_lens)

        if use_concat:
            self.sorted_data = x.data.new(sum(self.seq_lens), *x.size()[1:]).zero_()
            raise NotImplementedError()
        else:
            sorted_l = []
            for i, ind in enumerate(fwd_indices):
                seq_l = self.seq_lens[ind]
                sorted_l.append(x[:seq_l, ind])
            self.sorted_data = torch.cat(sorted_l)

    def as_packed(self):
        return PackedSequence(self.sorted_data, self.sorted_lens)

    def as_padded(self):
        return self.pad(self.as_packed())

    def pad(self, x):
        """
        Pads the PackedSequence x
        :param x:
        :return: [batch_size, T, :] array
        """
        out, lengths = pad_packed_sequence(x)
        out = out[self.perm].contiguous()
        return out, transpose_batch_sizes(lengths)

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