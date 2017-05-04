from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import random
import numpy as np
import numbers
from skimage.transform import resize
from skimage.io import ImageCollection, concatenate_images


def load_frames(folder_name, offset=0, desired_fps=3, max_frames=40):
    """
    :param folder_name: Filename with a gif
    :param offset: How many frames into the gif we want to start at
    :param desired_fps: How many fps we'll sample from the image
    :return: [T, h, w, 3] GIF
    """
    coll = ImageCollection(folder_name + '/out-*.jpg')

    try:
        duration_path = folder_name + '/duration.txt'
        with open(duration_path,'r') as f:
            durs = f.read().splitlines()
            fps = 100.0/durs[0]
    except:
        # Some error occurs
        fps = 10

    # want to scale it to desired_fps
    keep_ratio = max(1., fps/desired_fps)

    frames = np.arange(offset, len(coll), keep_ratio).astype(np.uint8)[:max_frames]
    print("Originally {} frames -> {} frames {} KR={}".format(len(coll), len(frames), frames, keep_ratio))

    return concatenate_images(coll[frames])


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

        print("Image shape is {}, size is {}".format(img.shape, self.size))

        if (w < tw) or (h < th):
            raise ValueError('Image too small')

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

        if (w < tw) or (h < th):
            raise ValueError('Image too small')

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
        return np.stack([resize(frame, (oh, ow), mode='constant') for frame in img])


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
        print("Vid shape is {}".format(vid.shape))
        img = torch.from_numpy(vid.transpose((0,3, 1, 2)))
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