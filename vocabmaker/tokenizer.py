"""
Default tokenizer used by vocabmaker
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import spacy
from spacy.symbols import ORTH, LEMMA, POS

from unidecode import unidecode

def _maybe_unicode(st):
    """
    TODO: make this python 3 compatible
    :param st:
    :return:
    """
    if not isinstance(st, unicode):
        st = unicode(st)
    st = unicode(unidecode(st))
    return st


INT_TO_STR = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six',7: 'seven',8: 'eight', 9: 'nine',
}


def catch_int(s):
    """
    Fixes numbers if they appear in text
    :param s: string that possibly contains a number
    :return: <NUM> if no number exists, else a text representation of that
                number
    """
    try:
        res = int(float(s))
        if res in INT_TO_STR:
            return INT_TO_STR[res]
        else:
            return '<NUM>'
    except ValueError:
        return None


def clean_tok(spacy_tok, filter_punct=True):
    """
    Applies some simple rules on text for tokenizing
    :param spacy_tok: input
    :param filter_punct: Whether to filter out punctuation
    :return:
    """
    if spacy_tok.like_num and not spacy_tok.is_ascii:
        return catch_int(spacy_tok.orth_.lower())
    if spacy_tok.is_space: # Filter out \n and \r.
        return None
    if filter_punct and spacy_tok.is_punct:
        return None
    return spacy_tok.orth_.lower()


class Tokenizer(object):
    """
    Default tokenizer. Uses spaCy under the hood
    """
    def __init__(self, special_cases=None):
        self.nlp = spacy.load('en')
        self.special_cases = special_cases
        self.add_special_cases()

    def add_special_cases(self):
        if self.special_cases is not None:
            for tok, pos in self.special_cases.iteritems():
                self.nlp.tokenizer.add_special_case(tok, [{ORTH: tok, LEMMA: tok, POS: pos}])

    def __getstate__(self):
        return {'special_cases': self.special_cases}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nlp = spacy.load('en')
        self.add_special_cases()

    def __call__(self, sent):
        return (clean_tok(t) for t in self.nlp(_maybe_unicode(sent)))
