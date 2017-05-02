"""
Tools for making a vocabulary.
Through the use of silly python hacks, vocab("a string") encodes it to numbers.
                    vocab[[list of numbers]] encodes it to a string!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import numpy as np
from vocabmaker.tokenizer import Tokenizer
import os
#
# PYTORCH_IMPORTED = True
# try:
#     # Optional pytorch
#     import torch
# except ImportError:
#     PYTORCH_IMPORTED = False



class Vocab(object):
    def __init__(self, vocab_size=65535, cutoff=5, tokenizer=None, compress_vocab=False,
                 eos = 0, bos=1, unk=2):
        """
        Class that handles the vocab that we'll create.
        :param vocab_size: Size of the vocab (we'll pick the vocab_size most frequent).
                           Default is 65535 (so we can fit it nicely in a 32-bit signed int).
        :param cutoff: A word must occur at least this many times for us to incorporate it.
               Default is 5. Can be None, in which case any word can be part of the vocab.
        :param tokenizer: Tokenizer used, if not the default spacy-based tokenizer.
                          Must be a callable argument that takes in a unicode string and returns

        :param compress_vocab: Only store the tokens that passed the threshold. Default: False
        :param eos: End-of-sequence token
        :param bos: Beginning-of-sequence token
        :param unk: Unknown token
        """

        if not callable(tokenizer):
            self.tokenizer = Tokenizer()

        self._vocab = []
        self._map = {}

        self.eos_token = eos
        self.bos_token = bos
        self.unk_token = unk
        self.vocab_size = vocab_size
        self.cutoff = cutoff

        self.compress_vocab = compress_vocab

        if vocab_size <= 65535:
            self.dtype = np.int32
        else:
            self.dtype = np.int64

    @classmethod
    def from_list(cls, l, vocab_size=None, tokenizer=None):
        """
        Makes a vocab from a list of words sorted by frequency.
        :param l: list of words, ignoring EOS, BOS, UNK, etc.
        :return:
        """
        if vocab_size is None:
            vocab_size = len(l) + 3
        v = cls(cutoff=None, vocab_size=vocab_size, tokenizer=tokenizer)
        v._vocab = [u"<EOS>", u"<BOS>", u"<UNK>"] + l
        v._make_map()
        return v

    def apply_sent(self, sent, max_len=-1):
        """
        Applies the dictionary to a sentence.
        :param sent: A string representing a sentence
        :param max_len: if -1, apply to the whole string, else apply to that
               many characters
        :return: 1d numpy array of tokens
        """
        assert self.is_loaded, "Vocab not loaded"

        if isinstance(sent, list):
            sent = ' '.join(sent)

        toks = [self.map(x) for x in self.tokenizer(sent) if x is not None]

        enc_tokens = np.array(toks, dtype=self.dtype)

        if max_len == -1:
            return enc_tokens
        return enc_tokens[:max_len]

    def apply_list(self, sents, max_len=-1, return_lengths=False):
        """
        Applies the dictionary to a list of sentences.
        :param sents: A list of strings
        :param max_len: if -1, apply to the whole string, else apply to that
           many characters
        :return: 2d numpy array of size [len(sents), max_len]. shorter strings
           get padded by EOS
        """
        assert self.is_loaded, "Vocab not loaded"

        enc_sents = [self.apply_sent(s, max_len) for s in sents]

        t_array = np.vstack([np.concatenate([x, np.full(max_len - x.shape[0], self.eos_token,
                                                        dtype=self.dtype)]) for x in enc_sents])

        if max_len == 1:
            t_array = np.squeeze(t_array)
        if return_lengths:
            return t_array, [len(s) for s in enc_sents]
        return t_array

    def reverse_apply_1d(self, toks):
        """
        Produces an english sentence from the array or list of tokens. Skips BOS
        :param toks: 1d numpy array of tokens
        :return: An english sentence (string)
        """
        assert self.is_loaded, 'vocab not loaded.'

        words = []
        for t in toks:
            if (len(words) == 0) and (t == self.bos_token):
                continue  # Skip BOS
            words.append(self._vocab[t])
            if t == self.eos_token:
                break

        return " ".join(words)

    def reverse_apply_list(self, toks):
        """
        Applies the dictionary to a number, or a list of numbers/array of numbers, etc..
        :param toks:
        :return:
        """
        assert self.is_loaded, "Vocab not loaded"
        return [self.reverse_apply_1d(t) for t in toks]

    def fit(self, sents):
        """
        Constructs a dictionary from training sents
        :param sents: a list/iterable of sentences, each in unicode format.
        """
        freq_dist = defaultdict(int)  # Apply tokenizer
        for plain_sent in sents:
            for t in self.tokenizer(plain_sent):
                freq_dist[t] += 1

        if None in freq_dist:
            freq_dist.pop(None)
        self.fit_from_dist(freq_dist)

    def fit_from_dist(self, freq_dist):
        """
        Fits the vocab from a dictionary of word -> count entries.
        :param freq_dist: Dictionary of word->count entries
        """
        # vocab will be a list of words, where the ind is the ID. omitting,
        self._vocab = [u"<EOS>", u"<BOS>", u"<UNK>"]

        counts = []
        for v, count in sorted(freq_dist.iteritems(), key=lambda (k, v): v, reverse=True):
            self._vocab.append(v)
            counts.append(count)

        if self.vocab_size is None:
            if self.cutoff is None:
                self.vocab_size = len(self._vocab)
            else:
                # searchsorted wants ascending order...
                self.vocab_size = np.searchsorted(-np.array(counts), -(self.cutoff-1))

        # Compress if desired
        if self.compress_vocab:
            self._vocab = self._vocab[:self.vocab_size]

        self._make_map()

    def __contains__(self, tok):
        """
        Returns true if word is in the dictionary with the given cutoff parameter
        :param tok:
        :return:
        """
        res = self.map(tok)
        return res != self.unk_token

    def save(self, fn):
        """
        Saves dictionary as a text file
        :param fn: where to save to
        """
        with open(fn, 'w') as f:
            for l in self._vocab:
                f.write(l + '\n')

    def load(self, fn):
        """
        Loads dictionary from a text file
        """
        if not os.path.exists(fn):
            raise IOError('Vocabulary file {} does not exist'.format(fn))

        with open(fn, 'r') as f:
            self._vocab = f.read().splitlines()

        assert len(self._vocab) > 3, "Vocab is empty"

        self._make_map()
        for t in [u"<EOS>", u"<BOS>", u"<UNK>"]:
            assert t in self._map, '%r not in dictionary' % t

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['_map']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._make_map()

    def _make_map(self):
        """
        Given a list where the index is the word ID and the value is the word,
        makes a mapping of the inverse
        """
        d = {w: i for i, w in enumerate(self._vocab)}
        self._map = defaultdict(lambda: self.unk_token, d)

    def __getitem__(self, key):
        """ Transform int to a word in dictionary """
        # assert isinstance(key, (int, long)), "Must be int"
        assert self.is_loaded, "Must be loaded"
        return self._vocab[key]

    def map(self, word):
        result = self._map[word]
        if result >= self.vocab_size:
            return self.unk_token
        return result

    def __call__(self, word):
        """ Transform a word to an ID"""
        return self.map(word)

    def __len__(self):
        return min(self.vocab_size, len(self._vocab))

    @property
    def is_loaded(self):
        vocab_len = len(self._vocab)
        map_len = len(self._map)

        # Map will prob. be larger because unk values are cached

        return (vocab_len <= map_len) and (vocab_len > 0)
