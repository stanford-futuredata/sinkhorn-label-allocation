#
# FixMatch algorithm
#
# Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang,
# and Colin Raffel. FixMatch: Simplifying semi-supervised learning with consistencyand confidence. In Advances in Neural
# Information Processing Systems, 2020
#

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from typing import Callable, Union, Generator, NamedTuple, Any, Sequence
from modules import Classifier, EMA
from utils import expand_generator, interleave, de_interleave, filter_parameters


def get_labeled_dist(dataset):
    counts = torch.unique(torch.tensor(dataset.targets), sorted=True, return_counts=True)[-1]
    return counts.float() / counts.sum()


class FixMatch(NamedTuple):
    num_workers: int  # number of workers per dataloader
    num_iters: int
    model_optimizer_ctor: Callable[..., torch.optim.Optimizer]
    lr_scheduler_ctor: Callable
    param_avg_ctor: Callable[..., EMA]
    labeled_batch_size: int
    unlabeled_batch_size: int
    unlabeled_weight: Union[float, Callable]
    threshold: float
    dist_alignment: bool  # whether to use the distribution alignment heuristic
    dist_alignment_batches: int  # number of batches used to estimate predicted label distribution
    dist_alignment_eps: float  # small float to avoid zero division
    mixed_precision: bool
    devices: Sequence[Union[torch.device, str]]

    class Stats(NamedTuple):
        iter: int
        loss: float
        loss_labeled: float
        loss_unlabeled: float
        model: Classifier
        avg_model: Classifier
        optimizer: torch.optim.Optimizer
        scheduler: Any
        threshold_frac: float  # fraction of examples in the batch above the confidence threshold

    def train(self, model: Classifier, labeled_dataset: Dataset, unlabeled_dataset: Dataset):
        return expand_generator(self.train_iter(model, labeled_dataset, unlabeled_dataset), return_only=True)

    def train_iter(
            self,
            model: Classifier,
            labeled_dataset: Dataset,
            unlabeled_dataset: Dataset) -> Generator[Stats, None, Any]:

        labeled_sampler = BatchSampler(RandomSampler(
            labeled_dataset, replacement=True, num_samples=self.num_iters*self.labeled_batch_size),
            batch_size=self.labeled_batch_size, drop_last=True)
        unlabeled_sampler = BatchSampler(RandomSampler(
            unlabeled_dataset, replacement=True, num_samples=self.num_iters*self.unlabeled_batch_size),
            batch_size=self.unlabeled_batch_size, drop_last=True)
        labeled_loader = DataLoader(
            labeled_dataset, batch_sampler=labeled_sampler, num_workers=self.num_workers, pin_memory=True)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=self.num_workers, pin_memory=True)

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

        if self.dist_alignment:
            labeled_dist = get_labeled_dist(labeled_dataset).to(self.devices[0])
            prev_labels = torch.full(
                [self.dist_alignment_batches, model.num_classes], 1 / model.num_classes, device=self.devices[0])
            prev_labels_idx = 0

        # training loop
        for batch_idx, (b_l, b_u) in enumerate(zip(labeled_loader, unlabeled_loader)):
            # labeled examples
            xl, yl = b_l
            yl = yl.cuda(non_blocking=True)

            # augmented pairs of unlabeled examples
            (xw, xs), _ = b_u

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                x = torch.cat([xl, xs, xw]).cuda(non_blocking=True)
                num_blocks = x.shape[0] // xl.shape[0]
                x = interleave(x, num_blocks)
                out = torch.nn.parallel.data_parallel(
                    model, x, module_kwargs={'autocast': self.mixed_precision}, device_ids=self.devices)
                out = de_interleave(out, num_blocks)

                # get labels
                with torch.no_grad():
                    probs = torch.softmax(out[-len(xw):], -1)
                    if self.dist_alignment:
                        model_dist = prev_labels.mean(0)
                        prev_labels[prev_labels_idx] = probs.mean(0)
                        prev_labels_idx = (prev_labels_idx + 1) % self.dist_alignment_batches
                        probs *= (labeled_dist + self.dist_alignment_eps) / (model_dist + self.dist_alignment_eps)
                        probs /= probs.sum(-1, keepdim=True)
                    yu = torch.argmax(probs, -1)
                    mask = (torch.max(probs, -1)[0] >= self.threshold).to(dtype=torch.float32)

                loss_l = F.cross_entropy(out[:len(xl)], yl, reduction='mean')
                loss_u = (mask * F.cross_entropy(out[len(xl):-len(xw)], yu, reduction='none')).mean()
                loss = loss_l + self.unlabeled_weight * loss_u

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
                loss_labeled=loss_l.cpu().item(),
                loss_unlabeled=loss_u.cpu().item(),
                model=model,
                avg_model=param_avg.avg_model,
                optimizer=optim,
                scheduler=scheduler,
                threshold_frac=mask.mean().cpu().item())
