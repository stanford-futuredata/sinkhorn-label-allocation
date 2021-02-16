#
# Image augmentation functions
#
# Adapted from:
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py

import random
import torch
import numpy as np
from PIL import ImageOps, ImageEnhance, Image


#
# Define RandAugment operations
#

PARAMETER_MAX = 10


def _float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def _int_parameter(level, maxval):
    return int(level * maxval / PARAMETER_MAX)


def identity(img: Image, level: int):
    return img


def autocontrast(img: Image, level: int):
    return ImageOps.autocontrast(img)


def equalize(img: Image, level: int):
    return ImageOps.equalize(img)


def rotate(img: Image, level: int):
    degrees = _int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees)


def posterize(img: Image, level: int):
    level = _int_parameter(level, 4)
    return ImageOps.posterize(img, 4 - level)


def shearx(img: Image, level: int):
    level = _float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def sheary(img: Image, level: int):
    level = _float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def translatex(img: Image, level: int):
    level = _int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translatey(img: Image, level: int):
    level = _int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def solarize(img: Image, level: int):
    level = _int_parameter(level, 256)
    return ImageOps.solarize(img, 256 - level)


def _enhancer_impl(enhancer):
    def impl(img, level):
        v = _float_parameter(level, 1.8) + .1
        return enhancer(img).enhance(v)
    return impl


def color(img: Image, level: int):
    return _enhancer_impl(ImageEnhance.Color)(img, level)


def contrast(img: Image, level: int):
    return _enhancer_impl(ImageEnhance.Contrast)(img, level)


def brightness(img: Image, level: int):
    return _enhancer_impl(ImageEnhance.Brightness)(img, level)


def sharpness(img: Image, level: int):
    return _enhancer_impl(ImageEnhance.Sharpness)(img, level)


augmentation_ops = [
    identity,
    autocontrast,
    equalize,
    rotate,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shearx,
    sheary,
    translatex,
    translatey,
    posterize,
]


class Maybe(object):
    def __init__(self, f, probability=0.5):
        self.f = f
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            img = self.f(img)
        return img


class RandAugment(object):
    def __init__(self, num_ops=2, num_levels=10, probability=0.5):
        self.num_ops = num_ops
        self.num_levels = num_levels
        self.probability = probability

    def __call__(self, img):
        for op in random.choices(augmentation_ops, k=self.num_ops):
            level = np.random.randint(1, self.num_levels)
            if random.random() < self.probability:
                img = op(img, level)
        return img


def cutout_tensor(img: torch.Tensor, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.
    The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
    This operation applies a `size`x`size` mask of zeros to a random location
    within `img`.
    Args:
      img: torch.Tensor image that cutout will be applied to.
      size: Height/width of the cutout mask that will be
    Returns:
      A tensor that is the result of applying the cutout mask to `img`.
    """
    if size <= 0:
        return img

    assert len(img.shape) == 3
    num_channels, img_height, img_width = img.shape
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    img = img.clone()
    img[:, upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1]] = 0
    return img
