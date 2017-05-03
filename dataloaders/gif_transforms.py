from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import math
import random
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import numbers
import types
import collections
from skimage.transform import resize
from torchvision.transforms import Compose
from scipy.ndimage.interpolation import zoom

def get_frames(img, keep_frames=None):
    """
    Gets frames from loaded GIF
    :return: [T, h, w, 3]
    """
    pal = img.getpalette()
    prev = img.convert('RGBA')
    prev_dispose = True

    if keep_frames is None:
        keep_frames = range(img.n_frames)
    keep_frames_set = set(keep_frames)

    all_frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        dispose = frame.dispose

        if frame.tile:
            x0, y0, x1, y1 = frame.tile[0][1]
            if not frame.palette.dirty:
                frame.putpalette(pal)
            frame = frame.crop((x0, y0, x1, y1))
            bbox = (x0, y0, x1, y1)
        else:
            bbox = None

        if dispose is None:
            prev.paste(frame, bbox, frame.convert('RGBA'))
            if i in keep_frames_set:
                all_frames.append(np.array(prev)[:,:,:3])
            prev_dispose = False
        else:
            if prev_dispose:
                prev = Image.new('RGBA', img.size, (0, 0, 0, 0))
            out = prev.copy()
            out.paste(frame, bbox, frame.convert('RGBA'))
            if i in keep_frames_set:
                all_frames.append(np.array(out)[:,:,:3])
    return np.stack(all_frames)


def load_gif(fn, offset=0, desired_fps=5):
    """
    :param fn: Filename with a gif
    :param offset: How many frames into the gif we want to start at
    :param desired_fps: How many fps we'll sample from the image
    :return: [T, h, w, 3] GIF
    """
    img = Image.open(fn)
    fps = 1000.0/img.info['duration']

    # want to scale it to desired_fps
    keep_ratio = min(1., fps/desired_fps)
    frames = np.arange(offset, img.n_frames, keep_ratio).astype(np.uint8)
    return get_frames(img, frames)


class RandomCrop(object):
    """Crops the given ndimage at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        assert self.padding == 0, "not implemented yet"

    def __call__(self, img):
        # if self.padding > 0:
        #     img = ImageOps.expand(img, border=self.padding, fill=0)
        h = img.shape[1]
        w = img.shape[2]

        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[:, y1:y1+th, x1:x1+tw]

class CenterCrop(object):
    """Crops the given ndimage at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        assert self.padding == 0, "not implemented yet"

    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]

        th, tw = self.size
        if w == tw and h == th:
            return img
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[:, y1:y1+th, x1:x1+tw]


class Scale(object):
    """Rescales the input numpy image to the given 'size'.
    Size will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)
        return np.stack([resize(frame, (oh, ow)) for frame in img])


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return np.stack([np.fliplr(x) for x in img])
        return img


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, vid):
        img = torch.from_numpy(pic.transpose((3, 1, 2)))
        return img.float().div(255)

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t in tensor:
            for c, m, s in zip(t, self.mean, self.std):
                c.sub_(m).div_(s)
        return tensor