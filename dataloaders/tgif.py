from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from vocabmaker.vocabmaker import Vocab

import csv
import os
import random

import pickle as pkl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import IntTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from unidecode import unidecode
from vocabmaker.tokenizer import _maybe_unicode
from dataloaders.gif_transforms import load_gif, RandomCrop, CenterCrop, ToTensor, Normalize, Scale

# Replace with your info
FN_TO_CAPS = 'dataloaders/tgif/tgif-v1.0.tsv'
GIFS_FOLDER = 'dataloaders/tgif/gifs'
SPLITS = 'dataloaders/tgif/splits'

# Needed for torch pretrained stuff
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _fix_fn(fn):
    """
    Makes it so that we're not referring to the tumblr.com URL
    """
    return '{}/{}'.format(GIFS_FOLDER, fn.split('/')[-1])


# Get the splits
def _read_split(ext):
    """
    Reads the train/test/val splits
    :param ext: string for "train", "val", or "test"
    :return:
    """
    with open('{}/{}.txt'.format(SPLITS, ext), 'r') as f:
        return [_fix_fn(fn) for fn in f.read().splitlines()]


def make_dataset(save_to='tgif-vocab.pkl'):
    """
    Constructs the TGIF dataset
    :return:    A list containing (fn, string) pairs for training
                A list containing (fn, string) pairs for validation
                A list containing (fn, string) pairs for testing
                A vocabulary of everything
    """
    if os.path.exists(save_to):
        with open(save_to, 'rb') as f:
            return pkl.load(f)

    splits = {m: set(_read_split(m)) for m in ('train','val','test')}

    # Get the data
    with open(FN_TO_CAPS,'r') as f:
        data_ = [(_fix_fn(fn), _maybe_unicode(cap))
                for fn, cap in csv.reader(f, delimiter='\t')]
        data = []
        for x in data_:
            # if not os.path.exists(x[0]):
            #     print("{} doesnt exist".format(x[0]))
            data.append(x)

    print("fitting vocab")
    vocab = Vocab(vocab_size=10000, compress_vocab=True)
    vocab.fit((x[1] for x in data if x[0] not in splits['test']))

    print("splitting")
    train = [d for d in data if d[0] in splits['train']]
    val = [d for d in data if d[0] in splits['val']]
    test = [d for d in data if d[0] in splits['test']]

    print("saving")
    with open(save_to, 'wb') as f:
        pkl.dump((train, val, test, vocab), f)
    return train, val, test, vocab


class TgifDataset(data.Dataset):
    def __init__(self, data, vocab, is_train=True, cnn_size=299):
        """
        :param all_sents: List of (input, target) sentences
        :param vocab: Vocab
        :param is_train:
        :param num_test:
        """
        self.data = data
        self.vocab = vocab
        self.is_train = is_train

        crop = RandomCrop if self.is_train else CenterCrop
        self.transform = transforms.Compose([
            Scale(int(cnn_size*1.1)),
            crop(cnn_size),
            ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):

        # start the video a little later
        offset = random.randint(0, 5) if self.is_train else 0
        vid = self.transform(load_gif(self.data[index][0], offset))
        target = IntTensor(self.vocab.apply_sent(self.data[index][1]))
        return vid, target

    def __len__(self):
        return len(self.data)


def collate_fn(data, bos_token, eos_token, cnn_size=299):
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

    # Stack over all frames and batch sizes
    in_lengths = [len(vid) for vid in inputs]
    in_data = torch.cat(inputs, 0) 

    # Pad outputs to being of the form (T, batch_size)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    out_lengths = [len(cap) + 1 for cap in outputs]
    out_data = torch.cat((
        (bos_token * torch.ones(1, batch_size)).long(),
        (eos_token * torch.ones(max(out_lengths), batch_size)).long(),
    ), 0)

    for i, (cap, end) in enumerate(zip(outputs, out_lengths)):
        out_data[1:end, i] = cap

    out_data_bos = pack_padded_sequence(out_data[:-1], out_lengths)
    out_data_eos = pack_padded_sequence(out_data[1:], out_lengths)

    return in_data, in_lengths, out_data_bos, out_data_eos


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
def loader(batch_size=32, use_test=False, shuffle=True, num_workers=1):
    train, val, test, vocab = make_dataset()

    t_data = TgifDataset(train, vocab)
    t_data_test = TgifDataset(val if not use_test else test, vocab, is_train=False)

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
