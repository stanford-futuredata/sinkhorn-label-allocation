import numpy as np
import logging
import torch
import modules
import utils
from fixmatch import FixMatch
from data import get_cifar10, get_cifar100, get_svhn
from monitoring import TableLogger
from evaluation import ModelEvaluator
from functools import partial
from tqdm import tqdm
import pprint
import datetime
import os
import re
import pickle
import fire


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
        num_workers=8,
        num_filters=32,
        dataset='cifar10',
        data_path='/tmp/data',
        output_dir='/tmp/fixmatch',
        run_id=None,
        num_labeled=40,
        seed=1,
        num_epochs=1024,
        batches_per_epoch=1024,
        checkpoint_interval=1024,
        max_checkpoints=25,
        snapshot_interval=None,
        optimizer='sgd',
        lr=0.03,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
        bn_momentum=1e-3,
        labeled_batch_size=64,
        unlabeled_batch_size=64*7,
        update_batch_size=64*7,
        unlabeled_weight=1.,
        exp_moving_avg_decay=1e-3,
        threshold=0.95,
        whiten=True,
        labeled_aug='weak',
        unlabeled_aug=('weak', 'strong'),
        sample_mode='label_dist_min1',
        dist_alignment=False,
        dist_alignment_batches=128,
        dist_alignment_eps=1e-6,
        mixed_precision=True,
        devices=('cuda:0',)):

    # initial setup
    num_batches = num_epochs * batches_per_epoch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = dict(locals())
    logger.info(pprint.pformat(args))

    run_id = datetime.datetime.now().isoformat() if run_id is None else run_id
    output_dir = os.path.join(output_dir, str(run_id))
    logger.info('output dir = %s' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    train_logger, eval_logger = TableLogger(), TableLogger()

    # load datasets
    if dataset == 'cifar10':
        dataset_fn = get_cifar10
    elif dataset == 'cifar100':
        dataset_fn = get_cifar100
    elif dataset == 'svhn':
        dataset_fn = get_svhn
    else:
        raise ValueError('Invalid dataset ' + dataset)
    datasets = dataset_fn(
        data_path, num_labeled, labeled_aug=labeled_aug, unlabeled_aug=unlabeled_aug,
        sample_mode=sample_mode, whiten=whiten)

    model = modules.WideResNet(
        num_classes=datasets['labeled'].num_classes, bn_momentum=bn_momentum, channels=num_filters)
    optimizer = partial(torch.optim.SGD, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    scheduler = partial(utils.WarmupCosineLrScheduler, warmup_iter=0, max_iter=num_batches)
    evaluator = ModelEvaluator(datasets['test'], labeled_batch_size + unlabeled_batch_size, num_workers)

    def evaluate(model, avg_model, iter):
        results = evaluator.evaluate(model, device=devices[0])
        avg_results = evaluator.evaluate(avg_model, device=devices[0])
        valid_stats = {
            'valid_loss': avg_results.log_loss,
            'valid_accuracy': avg_results.accuracy,
            'valid_loss_noavg': results.log_loss,
            'valid_accuracy_noavg': results.accuracy
        }
        eval_logger.write(
            iter=iter,
            **valid_stats)
        eval_logger.step()
        return avg_results.accuracy

    def checkpoint(model, avg_model, optimizer, scheduler, iter, fmt='ckpt-{:08d}.pt'):
        path = os.path.join(output_dir, fmt.format(iter))
        torch.save(dict(
            iter=iter,
            model=model.state_dict(),
            avg_model=avg_model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict()), path)
        checkpoint_files = sorted(list(filter(lambda x: re.match(r'^ckpt-[0-9]+.pt$', x), os.listdir(output_dir))))
        if len(checkpoint_files) > max_checkpoints:
            os.remove(os.path.join(output_dir, checkpoint_files[0]))
        train_logger.to_dataframe().to_pickle(os.path.join(output_dir, 'train.log.pkl'))
        eval_logger.to_dataframe().to_pickle(os.path.join(output_dir, 'eval.log.pkl'))

    trainer = FixMatch(
        num_iters=num_epochs * batches_per_epoch,
        num_workers=num_workers,
        model_optimizer_ctor=optimizer,
        lr_scheduler_ctor=scheduler,
        param_avg_ctor=partial(modules.EMA, alpha=exp_moving_avg_decay),
        labeled_batch_size=labeled_batch_size,
        unlabeled_batch_size=unlabeled_batch_size,
        unlabeled_weight=unlabeled_weight,
        threshold=threshold,
        dist_alignment=dist_alignment,
        dist_alignment_batches=dist_alignment_batches,
        dist_alignment_eps=dist_alignment_eps,
        mixed_precision=mixed_precision,
        devices=devices)

    timer = utils.Timer()
    with tqdm(desc='train', total=num_batches, position=0) as train_pbar:
        train_iter = utils.Generator(
            trainer.train_iter(model, datasets['labeled'], datasets['unlabeled']))
        smoothed_loss = utils.ema(0.3, avg_only=True)
        smoothed_loss.send(None)
        smoothed_acc = utils.ema(1., avg_only=False)
        smoothed_acc.send(None)
        eval_stats = None, None

        # training loop
        for i, stats in enumerate(train_iter):
            train_pbar.set_postfix(
                loss=smoothed_loss.send(stats.loss), eval_acc=eval_stats[0], eval_v=eval_stats[1], refresh=False)
            train_pbar.update()
            train_logger.write(
                loss=stats.loss, loss_labeled=stats.loss_labeled, loss_unlabeled=stats.loss_unlabeled,
                threshold_frac=stats.threshold_frac, time=timer())

            if (checkpoint_interval is not None
                and i > 0 and (i+1) % checkpoint_interval == 0) or (i == num_batches - 1):
                eval_acc = evaluate(stats.model, stats.avg_model, i+1)
                eval_stats = smoothed_acc.send(eval_acc)
                checkpoint(stats.model, stats.avg_model, stats.optimizer, stats.scheduler, i+1)
                logger.info('eval acc = %.4f | allocated frac = %.4f' % (eval_acc, stats.threshold_frac))

            # take snapshots that are guaranteed to be preserved
            if snapshot_interval is not None and i > 0 and (i+1) % snapshot_interval == 0:
                checkpoint(stats.model, stats.avg_model, stats.optimizer,
                           stats.scheduler, i+1, 'snapshot-{:08d}.pt')

            train_logger.step()


if __name__ == '__main__':
    fire.Fire(main)
