#
# Supervised training
#

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from typing import Callable, Union, Generator, NamedTuple, Any, Sequence
from modules import Classifier, EMA
from utils import expand_generator, filter_parameters


class Supervised(NamedTuple):
    model_optimizer_ctor: Callable[..., torch.optim.Optimizer]
    lr_scheduler_ctor: Callable
    param_avg_ctor: Callable[..., EMA]
    num_iters: int
    batch_size: int
    num_workers: int  # number of workers per dataloader
    mixed_precision: bool
    devices: Sequence[Union[torch.device, str]]

    class Stats(NamedTuple):
        iter: int
        loss: float
        model: Classifier
        avg_model: Classifier
        optimizer: torch.optim.Optimizer
        scheduler: Any

    def train(self, model: Classifier, dataset: Dataset):
        return expand_generator(self.train_iter(model, dataset), return_only=True)

    def train_iter(
            self,
            model: Classifier,
            dataset: Dataset) -> Generator[Stats, None, Any]:

        sampler = BatchSampler(RandomSampler(
            dataset, replacement=True, num_samples=self.num_iters*self.batch_size),
            batch_size=self.batch_size, drop_last=True)
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=self.num_workers,
            worker_init_fn=lambda i: np.random.seed(torch.initial_seed() % 2**32 + i),
            pin_memory=True)

        model.to(device=self.devices[0])
        param_avg = self.param_avg_ctor(model)

        # set up optimizer without weight decay on batch norm or bias parameters
        no_wd_filter = lambda m, k: isinstance(m, nn.BatchNorm2d) or k.endswith('bias')
        wd_filter = lambda m, k: not no_wd_filter(m, k)
        optim = self.model_optimizer_ctor([
            {'params': filter_parameters(model, wd_filter)},
            {'params': filter_parameters(model, no_wd_filter), 'weight_decay': 0.}
        ])

        scheduler = self.lr_scheduler_ctor(optim)
        scaler = torch.cuda.amp.GradScaler()

        # training loop
        for batch_idx, (x, y) in enumerate(loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                out = torch.nn.parallel.data_parallel(
                    model, x, module_kwargs={'autocast': self.mixed_precision}, device_ids=self.devices)
                loss = F.cross_entropy(out, y, reduction='mean')

            model.zero_grad()
            if self.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            param_avg.step()
            scheduler.step()

            yield self.Stats(
                iter=batch_idx+1,
                loss=loss.cpu().item(),
                model=model,
                avg_model=param_avg.avg_model,
                optimizer=optim,
                scheduler=scheduler)
