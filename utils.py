import numpy as np
import torch
import torch.nn as nn
import time
import os
import math
from typing import TypeVar, Generator, Union, Sequence, Tuple, Callable, NamedTuple
import logging


logger = logging.getLogger(__name__)


def interleave(x: torch.Tensor, bt: int):
    """Rearrange a tensor of the form [1..1;2..2;...;bt..bt] into [12..bt;12..bt;...]."""
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x: torch.Tensor, bt: int):
    """The inverse of interleave()."""
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])


def filter_parameters(module: nn.Module, condition: Callable[[nn.Module, str], bool]):
    """A generator that yields parameters satisfying a given predicate."""
    params = set()
    for module_key, parent in module.named_modules():
        for param_key, param in parent.named_parameters(recurse=False):
            if param not in params and condition(parent, param_key):
                params.add(param)
                yield param


def get_mean_and_std(dataset):
    """Compute the per-channel mean and standard deviation of an image dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std.')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def urandom():
    int.from_bytes(os.urandom(4), 'little')


def to_numpy(x):
    return x.detach().cpu().numpy()


def ema(alpha=0.01, avg_only=True):
    """An exponential moving average generator."""
    avg, vel, var, std = None, None, None, None
    while True:
        if avg_only:
            x = yield avg
        else:
            x = (yield avg, vel, std)
        if avg is None:
            avg = x
            vel = var = std = 0.
        else:
            delta = x - avg
            vel = alpha * delta
            avg += vel
            var = (1 - alpha) * (var + alpha * delta**2)
            std = math.sqrt(var)


# From //github.com/davidcpage/cifar10-fast/bag_of_tricks.ipynb
class Timer(object):
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t


YieldType = TypeVar('YieldType')
ReturnType = TypeVar('ReturnType')
T = Tuple[Sequence[YieldType], ReturnType]


def expand_generator(g: Generator[YieldType, None, ReturnType], return_only: bool = False) -> Union[T, ReturnType]:
    """Given a finite generator that yields values (a_1, a_2, ..., a_T) and returns value b, returns the tuple
    ((a_1, ..., a_T), b), or only b if `return_only` is True."""
    ret = None

    def _():
        nonlocal ret
        ret = yield from g

    vals = tuple(_())
    if return_only:
        return ret
    return vals, ret


class Generator(object):
    def __init__(self, g):
        self.g = g
        self.value = None

    def __iter__(self):
        self.value = yield from self.g


# Cosine learning rate scheduler.
#
# From https://github.com/valencebond/FixMatch_pytorch/blob/master/lr_scheduler.py
class WarmupCosineLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class PiecewiseLinear(NamedTuple):
    knots: Sequence[float]
    vals: Sequence[float]

    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
