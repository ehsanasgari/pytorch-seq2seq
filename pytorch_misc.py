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
    mask = Variable(torch.zeros(len(context_lens), max(context_lens)))
    if torch.cuda.is_available():
        mask = mask.cuda()

    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1
    return mask


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



def pad_unsorted_sequence(sequences, lengths):
    """
    Pads the sequences that is not necessarily in longest-batch-first order
    :param sequences: A (B*t,:) tensor
    :param lengths: The lengths of the sequences
    :return: A (max T, B, :) tensor, and also permutation indices to return to normal
    unsorted order
    """
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if True, the output will be in BxTx* format.

    Returns:
        Tuple of Variable containing the padded sequence, and a list of lengths
        of each sequence in the batch.
    """
    sorted_lengths, fwd_indices = torch.sort(torch.IntTensor(lengths), dim=0, descending=True)
    inv_indices = torch.sort(fwd_indices)[1]

    print("Lengths: {}".format(lengths))
    print("Sorted lengths: {}".format(sorted_lengths))
    print("perm_indices {}".format(inv_indices))

    batch_size = len(lengths)
    max_t = sorted_lengths[0]
    output = sequences.data.new(max_t, batch_size, *sequences.size()[1:]).zero_()
    output = Variable(output)

    data_offset = 0
    for seq_l, sorted_seq_id in zip(lengths, inv_indices):
        output[:seq_l, sorted_seq_id] = sequences[data_offset:data_offset + seq_l]

    if torch.cuda.is_available():
        inv_indices = inv_indices.cuda()
        print("Hi {}".format(inv_indices))

    return output, inv_indices
