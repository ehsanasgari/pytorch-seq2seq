"""
Does bag of words, concatenating all definitions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pkl
import os

import numpy as np
import spacy
import torch
import torch.utils.data as data
from torch import IntTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from unidecode import unidecode

from vocabmaker.vocabmaker import Vocab


def make_dataset(text_file='dataloaders/janeausten.txt', save_to='dataloaders/data.pkl', max_len=20,
                 reverse_target=False):
    """
    Makes dataset for "translation"; ie, a basic seq2seq autoencoder
    :param text_file: Text file containing a bunch of sentences
    :param save_to: Where to cache the data
    :param max_len: Max length per sequence
    :param reverse_target: Whether or not we want the target sequences reversed
    :return:
    """
    if os.path.exists(save_to):
        with open(save_to, 'rb') as f:
            return pkl.load(f)

    with open(text_file, 'r') as f:
        corpus = ' '.join([unidecode(x.decode('utf-8')) for x in f.read().splitlines()])
    nlp = spacy.load('en')

    sents = [[y.orth_ for y in x if y.is_alpha] for x in nlp(corpus).sents]
    vocab = Vocab(vocab_size=5000, compress_vocab=True)
    vocab.fit([' '.join(s) for s in sents if len(s) != 0])

    np.random.seed(123456)

    # Optional map if you want to do something here.
    # vocab_permute = np.concatenate((np.arange(3), np.random.permutation(4997)+3))
    # old_to_new = {vocab._vocab[i]: vocab._vocab[j] for i,j in enumerate(vocab_permute)}
    old_to_new = {v: v for v in vocab._vocab}

    all_sents = []
    for s_raw in sents:
        res = [(x, old_to_new[x]) for x in s_raw if x in old_to_new][:max_len]
        if len(res) > 0:
            s, s_perm = zip(*res)
            if reverse_target:
                all_sents.append((' '.join(s), ' '.join(s_perm[::-1])))
            all_sents.append((' '.join(s), ' '.join(s_perm)))

    with open(save_to, 'wb') as f:
        pkl.dump((all_sents, vocab), f)
    return all_sents, vocab


class TranslationDataset(data.Dataset):
    def __init__(self, all_sents, vocab, is_train=True, num_test=100):
        """
        :param all_sents: List of (input, target) sentences
        :param vocab: Vocab
        :param is_train:
        :param num_test:
        """
        self.all_sents = all_sents
        self.vocab = vocab
        self.is_train = is_train
        self.num_test = num_test
        self.offset = num_test if is_train else 0

    def __getitem__(self, index):
        i = index + self.offset
        input = IntTensor(self.vocab.apply_sent(self.all_sents[i][0]))
        target = IntTensor(self.vocab.apply_sent(self.all_sents[i][1]))

        return input, target

    def __len__(self):
        if self.is_train:
            return len(self.all_sents) - self.num_test
        else:
            return self.num_test


def collate_fn(data, bos_token, eos_token):
    """
    Creates minibatch tensors from list of (input, output) pairs

    :param data: tuple of input, output. Both are of the same length.
    :return: PackedSequence for the input sequence
             PackedSequence for the output sequence (starting with BOS)
             PackedSequence for the output sequence (ending with EOS)
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    inputs, outputs = zip(*data)

    batch_size = len(inputs)

    # Pad inputs
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    in_lengths = [len(cap) for cap in inputs]
    in_data = torch.zeros(max(in_lengths), batch_size).long()
    for i, (cap, end) in enumerate(zip(inputs, in_lengths)):
        in_data[:end, i] = cap[:end]

    # Pad outputs to being of the form (T, batch_size)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    out_lengths = [len(cap) + 1 for cap in outputs]
    out_data = torch.cat((
        (bos_token * torch.ones(1, batch_size)).long(),
        (eos_token * torch.ones(max(out_lengths), batch_size)).long(),
    ), 0)

    for i, (cap, end) in enumerate(zip(outputs, out_lengths)):
        out_data[1:end, i] = cap

    in_data = pack_padded_sequence(in_data, in_lengths)
    out_data_bos = pack_padded_sequence(out_data[:-1], out_lengths)
    out_data_eos = pack_padded_sequence(out_data[1:], out_lengths)

    return in_data, out_data_bos, out_data_eos


class MegaDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, but also loads everything as a (cuda) variable
    """

    @staticmethod
    def _load(item):
        def _cudaize(t):
            data = Variable(t.data)
            if torch.cuda.is_available():
                data = data.cuda()
            return PackedSequence(data, t.batch_sizes)

        return [_cudaize(t) for t in item]

    def __iter__(self):
        return (self._load(x) for x in super(MegaDataLoader, self).__iter__())


# Handles the loading
def loader(batch_size=32, shuffle=True, num_workers=1):
    all_sents, vocab = make_dataset()
    t_data = TranslationDataset(all_sents, vocab)
    t_data_test = TranslationDataset(all_sents, vocab, is_train=False)

    train_data_loader = MegaDataLoader(
        dataset=t_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, t_data.vocab.bos_token, t_data.vocab.eos_token),
    )

    test_data_loader = MegaDataLoader(
        dataset=t_data_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Load data in the main process
        collate_fn=lambda x: collate_fn(x, t_data.vocab.bos_token, t_data.vocab.eos_token)
    )

    return vocab, train_data_loader, test_data_loader
