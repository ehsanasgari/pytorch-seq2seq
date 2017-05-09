from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import os
import random

import spacy

import dill as pkl
import torch
from dataloaders.text import torchtext

import torchvision.transforms as transforms
from torch import IntTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from pytorch_misc import pad_list
from dataloaders.gif_transforms import load_frames, RandomCrop, CenterCrop, ToTensor, Normalize, Scale

# Replace with your info
FN_TO_CAPS = 'dataloaders/tgif/tgif-v1.0.tsv'
GIFS_FOLDER = 'dataloaders/tgif/jpgs/'
SPLITS = 'dataloaders/tgif/splits'

# Needed for torch pretrained stuff
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _fix_fn(fn):
    """Input: tumblr URL,
    output: folder of JPG images, ie dataloaders/tgif/jpgs/tumblr_lltznfhAdJ1qdoyv4o2_r1_500/
    """
    return '{}/{}'.format(GIFS_FOLDER, fn.split('/')[-1].split('.')[0])


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
    caps_field = torchtext.data.Field(
        tokenize='spacy',
        init_token='<bos>',
        eos_token='<eos>',
        lower=True,
    )

    if os.path.exists(save_to):
        with open(save_to, 'rb') as f:
            train, val, vocab = pkl.load(f)
            caps_field.vocab = vocab
            return train, val, caps_field

    splits = {m: set(_read_split(m)) for m in ('train','val','test')}

    # Get the data
    with open(FN_TO_CAPS,'r') as f:
        data_ = [(_fix_fn(fn), cap) for fn, cap in csv.reader(f, delimiter='\t')]
        data = []
        for x in data_:
            if os.path.exists(x[0]):
                data.append(x)

    caps_field.build_vocab([(x[1] for x in data if x[0] not in splits['test'])], max_size=10000)

    train = [d for d in data if d[0] in splits['train']]
    val = [d for d in data if d[0] in splits['val']]
    # test = [d for d in data if d[0] in splits['test']]

    print("saving")
    with open(save_to, 'wb') as f:
        pkl.dump((train, val, caps_field.vocab), f)
    return train, val, caps_field


class TgifDataset(torch.utils.data.Dataset):
    def __init__(self, data, caps_field, is_train=True, cnn_size=224, scale_size=256):
        """
        :param all_sents: List of (input, target) sentences
        :param vocab: Vocab
        :param is_train:
        :param num_test:
        """
        self.data = data
        self.caps_field = caps_field
        self.is_train = is_train

        crop = RandomCrop if self.is_train else CenterCrop
        self.transform = transforms.Compose([
            Scale(scale_size),
            crop(cnn_size),
            ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):

        # start the video a little later
        offset = random.randint(0, 5) if self.is_train else 0
        vid = self.transform(load_frames(self.data[index][0], offset))
        target = self.caps_field.preprocess(self.data[index][1])

        return vid, target

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """
    Creates minibatch tensors from list of (input, output) pairs

    :param data: tuple of input, output.
    """
    # We expect inputs to be longer, so well have those as the unpadded
    data.sort(key=lambda x: len(x.vid), reverse=True)
    vids, caps = zip(*data)

    in_data_pad, in_lens = pad_list(vids)
    in_data = pack_padded_sequence(in_data_pad, in_lens)

    out_data_pad, out_lens = pad_list(caps)
    return in_data, out_data_pad, out_lens


class CudaDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, but also loads everything as a (cuda) variable
    """
    @staticmethod
    def _load(item):
        def _cudaize_packed(t):
            data = Variable(t.data)
            if torch.cuda.is_available():
                data = data.cuda()
            return PackedSequence(data, t.batch_sizes)

        def _cudaize(t):
            data = Variable(t)
            if torch.cuda.is_available():
                data = data.cuda()
            return data

        return _cudaize_packed(item[0]), _cudaize(item[1]), item[2]

    def __iter__(self):
        return (self._load(x) for x in super(CudaDataLoader, self).__iter__())


# Handles the loading
def loader(batch_size=32, shuffle=True, num_workers=1):
    train, val, vocab = make_dataset()

    t_data = TgifDataset(train, vocab)
    t_data_test = TgifDataset(val, vocab, is_train=False)

    train_data_loader = CudaDataLoader(
        dataset=t_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_data_loader = CudaDataLoader(
        dataset=t_data_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Load data in the main process
        collate_fn=collate_fn
    )

    return train_data_loader, test_data_loader, vocab,
