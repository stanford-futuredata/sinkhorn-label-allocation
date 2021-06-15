#
# Self-training with Sinkhorn Label Allocation
#

import numpy as np
import logging
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from typing import Callable, Union, Generator, NamedTuple, Any, Sequence
from modules import Classifier, EMA
from utils import expand_generator, interleave, de_interleave, filter_parameters
import scipy.stats


logger = logging.getLogger(__name__)


class SinkhornLabelAllocation(object):
    cost_matrix: torch.Tensor
    log_Q: torch.Tensor  # log assignment matrix
    u: torch.Tensor  # row scaling variables
    v: torch.Tensor  # column scaling variables
    log_upper_bounds: torch.Tensor  # log class upper bounds
    rho: float  # allocation fraction
    reg: float  # regularization coefficient
    update_tol: float

    def __init__(
            self,
            num_examples: int,
            log_upper_bounds: torch.Tensor,
            allocation_param: float,
            reg: float,
            update_tol: float,
            device='cpu'):
        self.num_examples = num_examples
        self.num_classes = len(log_upper_bounds)
        self.cost_matrix = torch.zeros(self.num_examples + 1, self.num_classes + 1, device=device)
        self.u = torch.zeros(self.num_examples + 1, device=device)
        self.v = torch.zeros(self.num_classes + 1, device=device)
        self.log_upper_bounds = log_upper_bounds
        self.upper_bounds = torch.exp(log_upper_bounds).to(device)
        self.reg = reg
        self.update_tol = update_tol
        self.set_allocation_param(allocation_param)
        self.reset()

    def reset(self):
        self.u.zero_()
        self.v.zero_()
        self.cost_matrix[:-1, :-1] = np.log(self.num_classes)
        self.log_Q = F.log_softmax(-self.reg * self.cost_matrix, -1)

    def get_plan(self, idxs=None, log_p=None):
        assert (idxs is None or log_p is None)
        if idxs is None and log_p is None:
            return self.log_Q.exp()
        elif idxs is not None:
            return self.log_Q[idxs].exp()
        else:
            z = self.v.repeat(log_p.shape[0], 1)
            z[:, :-1] += self.reg * log_p
            return F.softmax(z, 1)

    def get_assignment(self, idxs=None, log_p=None):
        assert(idxs is None or log_p is None)
        if idxs is None and log_p is None:
            return torch.argmax(self.log_Q[:-1], 1)
        elif idxs is not None:
            return torch.argmax(self.log_Q[idxs], 1)
        else:
            z = self.v.repeat(log_p.shape[0], 1)
            z[:, :-1] += self.reg * log_p
            return torch.argmax(z, 1)

    def set_allocation_param(self, val: float):
        self.rho = val
        return self

    def set_cost_matrix(self, cost_matrix: torch.Tensor):
        self.cost_matrix.copy_(cost_matrix)
        self.log_Q = -self.reg * self.cost_matrix + self.u.view(-1, 1) + self.v.view(1, -1)
        return self

    def update_cost_matrix(self, log_p: torch.Tensor, idxs: torch.LongTensor):
        self.cost_matrix[idxs, :-1] = -log_p.detach()
        log_Q = -self.reg * self.cost_matrix[idxs] + self.v.view(1, -1)
        self.u[idxs] = -torch.logsumexp(log_Q, 1)
        self.log_Q[idxs] = log_Q + self.u[idxs].view(-1, 1)
        return self

    def update(self):
        mat = -self.reg * self.cost_matrix
        iters = 0
        mu = 1 - self.upper_bounds.sum()
        rn = 1 + self.num_classes + self.num_examples * (1 - self.rho - mu.clamp(max=0))
        c = torch.cat([
            1 + self.num_examples * self.upper_bounds,
            1 + self.num_examples * (1 - self.rho + mu.clamp(min=0).view(-1))])

        err = np.inf
        while err >= self.update_tol:
            # update columns
            log_Q = mat + self.u.view(-1, 1)
            self.v = torch.log(c) - torch.logsumexp(log_Q, 0)
            self.v -= self.v[:-1].mean()

            # update rows
            log_Q = mat + self.v.view(1, -1)
            self.u = -torch.logsumexp(log_Q, 1)
            self.u[-1] += torch.log(rn)
            self.log_Q = log_Q + self.u.view(-1, 1)

            err = (torch.abs(self.log_Q.exp().sum(0) - c).sum() / c.sum()).cpu().item()
            iters += 1

        return err, iters


# Excerpt from https://www.statsmodels.org/stable/_modules/statsmodels/stats/proportion.html#proportion_confint
def wilson_confint(count, nobs, alpha=0.05):
    """Wilson confidence interval for a binomial proportion"""
    count = np.asarray(count)
    nobs = np.asarray(nobs)
    q_ = count * 1. / nobs
    crit = scipy.stats.norm.isf(alpha / 2.)
    crit2 = crit ** 2
    denom = 1 + crit2 / nobs
    center = (q_ + crit2 / (2 * nobs)) / denom
    dist = crit * np.sqrt(q_ * (1. - q_) / nobs + crit2 / (4. * nobs ** 2))
    dist /= denom
    ci_low = center - dist
    ci_upp = center + dist
    return ci_low, ci_upp


def get_log_upper_bounds(dataset, method='empirical', **kwargs):
    counts = torch.unique(torch.tensor(dataset.targets), sorted=True, return_counts=True)[-1]
    num_examples = counts.sum()
    if method == 'none':
        return torch.zeros_like(counts)
    if method == 'empirical':
        return torch.log(counts.float() / counts.sum())
    if method == 'wilson':
        return torch.log(torch.tensor([wilson_confint(c, num_examples, kwargs['alpha'])[1] for c in counts]))
    raise ValueError('Invalid method: {}'.format(method))


class SLASelfTraining(NamedTuple):
    model_optimizer_ctor: Callable[..., torch.optim.Optimizer]
    lr_scheduler_ctor: Callable
    param_avg_ctor: Callable[..., EMA]
    allocation_schedule: Callable[[float], float]
    num_iters: int
    labeled_batch_size: int
    unlabeled_batch_size: int
    unlabeled_weight: Union[float, Callable]
    sinkhorn_reg: float
    update_tol: float
    upper_bound_method: str
    upper_bound_kwargs: dict
    num_workers: int
    mixed_precision: bool
    devices: Sequence[Union[torch.device, str]]

    class Stats(NamedTuple):
        iter: int
        loss: float
        loss_labeled: float
        loss_unlabeled: float
        allocation_param: float
        label_vars: torch.Tensor
        scaling_vars: torch.Tensor
        model: Classifier
        avg_model: Classifier
        optimizer: torch.optim.Optimizer
        scheduler: Any
        assgn_err: float
        assgn_iters: int

    def train(self, model: Classifier, labeled_dataset: Dataset, unlabeled_dataset: Dataset):
        return expand_generator(self.train_iter(model, labeled_dataset, unlabeled_dataset), return_only=True)

    def train_iter(
            self,
            model: Classifier,
            labeled_dataset: Dataset,
            unlabeled_dataset: Dataset) -> Generator[Stats, None, Any]:

        labeled_sampler = BatchSampler(RandomSampler(
            labeled_dataset, replacement=True, num_samples=self.num_iters * self.labeled_batch_size),
            batch_size=self.labeled_batch_size, drop_last=True)
        unlabeled_sampler = BatchSampler(RandomSampler(
            unlabeled_dataset, replacement=True,
            num_samples=self.num_iters * self.unlabeled_batch_size),
            batch_size=self.unlabeled_batch_size, drop_last=True)
        labeled_loader = DataLoader(
            labeled_dataset, batch_sampler=labeled_sampler, num_workers=self.num_workers,
            worker_init_fn=lambda i: np.random.seed(torch.initial_seed() % 2 ** 32 + i),
            pin_memory=True)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=self.num_workers,
            worker_init_fn=lambda i: np.random.seed(torch.initial_seed() % 2 ** 32 + self.num_workers + i),
            pin_memory=True)

        # initialize model and optimizer
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

        # initialize label assignment
        log_upper_bounds = get_log_upper_bounds(
            labeled_dataset, method=self.upper_bound_method, **self.upper_bound_kwargs)
        logger.info('upper bounds = {}'.format(torch.exp(log_upper_bounds)))
        label_assgn = SinkhornLabelAllocation(
            num_examples=len(unlabeled_dataset),
            log_upper_bounds=log_upper_bounds,
            allocation_param=0.,
            reg=self.sinkhorn_reg,
            update_tol=self.update_tol,
            device=self.devices[0])

        # training loop
        for batch_idx, (b_l, b_u) in enumerate(zip(labeled_loader, unlabeled_loader)):
            # labeled examples
            xl, yl = b_l
            yl = yl.cuda(non_blocking=True)

            # augmented pairs of unlabeled examples
            (xu1, xu2), idxs = b_u

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                x = torch.cat([xl, xu1, xu2]).cuda(non_blocking=True)
                if len(self.devices) > 1:
                    num_blocks = x.shape[0] // xl.shape[0]
                    x = interleave(x, num_blocks)
                    out = torch.nn.parallel.data_parallel(
                        model, x, module_kwargs={'autocast': self.mixed_precision}, device_ids=self.devices)
                    out = de_interleave(out, num_blocks)
                else:
                    out = model(x, autocast=self.mixed_precision)

                # compute labels
                logp_u = F.log_softmax(out[len(xl):], -1)
                nu = logp_u.shape[0] // 2
                qu = label_assgn.get_plan(log_p=logp_u[:nu].detach()).to(dtype=torch.float32, device=out.device)
                qu = qu[:, :-1]

                # compute loss
                loss_l = F.cross_entropy(out[:len(xl)], yl, reduction='mean')
                loss_u = -(qu * logp_u[nu:]).sum(-1).mean()
                loss = loss_l + self.unlabeled_weight * loss_u

                # update plan
                rho = self.allocation_schedule((batch_idx + 1) / self.num_iters)
                label_assgn.set_allocation_param(rho)
                label_assgn.update_cost_matrix(logp_u[:nu], idxs)
                assgn_err, assgn_iters = label_assgn.update()

            optim.zero_grad()
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
                iter=batch_idx + 1,
                loss=loss.cpu().item(),
                loss_labeled=loss_l.cpu().item(),
                loss_unlabeled=loss_u.cpu().item(),
                model=model,
                avg_model=param_avg.avg_model,
                allocation_param=rho,
                optimizer=optim,
                scheduler=scheduler,
                label_vars=qu,
                scaling_vars=label_assgn.v.data,
                assgn_err=assgn_err,
                assgn_iters=assgn_iters)
