import numpy as np
import logging
import torch
import modules
import utils
from supervised import Supervised
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
        dataset='cifar10',
        data_path='/tmp/data',
        output_dir='/tmp/supervised',
        run_id=None,
        seed=1,
        block_depth=4,
        num_filters=32,
        num_epochs=1024,
        batches_per_epoch=1024,
        batch_size=512,
        lr=0.03,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
        bn_momentum=1e-3,
        exp_moving_avg_decay=1e-3,
        augmentation='strong',
        checkpoint_interval=1024,
        max_checkpoints=25,
        num_workers=4,
        mixed_precision=True,
        devices=('cuda:0',)):
    """Supervised training.

    Args:
      dataset: the dataset to use ('cifar10', 'cifar100', 'svhn')
      data_path: dataset root directory
      output_dir: directory to save logs and model checkpoints
      run_id: name for training run (output will be saved under output_dir/run_id)
      seed: random seed
      block_depth: WideResNet block depth
      num_filters: WideResNet base filter count
      num_epochs: number of training epochs
      batches_per_epoch: number of batches per epoch
      batch_size: number of examples per batch
      lr: SGD initial learning rate
      momentum: SGD momentum parameter
      nesterov: whether to use SGD with Nesterov acceleration
      weight_decay: weight decay parameter
      bn_momentum: batch normalization momentum parameter
      exp_moving_avg_decay: model parameter exponential moving average decay
      augmentation: data augmentation mode ('none', 'weak', 'strong', 'weak_noflip', 'strong_noflip').
        'strong' augmentation uses RandAugment. 'noflip' disables horizontal flip augmentation.
      checkpoint_interval: number of batches between checkpoints
      max_checkpoints: maximum number of checkpoints to retain
      num_workers: number of workers per data loader
      mixed_precision: whether to use mixed precision training
      devices: list of devices for data parallel training
    """

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
        data_path, num_labeled=None, labeled_aug=augmentation, whiten=True)

    model = modules.WideResNet(
        num_classes=datasets['labeled'].num_classes, bn_momentum=bn_momentum,
        block_depth=block_depth, channels=num_filters)
    optimizer = partial(torch.optim.SGD, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    scheduler = partial(utils.WarmupCosineLrScheduler, warmup_iter=0, max_iter=num_batches)
    evaluator = ModelEvaluator(datasets['test'], batch_size, num_workers)
    param_avg_ctor = partial(modules.EMA, alpha=exp_moving_avg_decay)

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

    trainer = Supervised(
        num_iters=num_epochs * batches_per_epoch,
        num_workers=num_workers,
        model_optimizer_ctor=optimizer,
        lr_scheduler_ctor=scheduler,
        param_avg_ctor=param_avg_ctor,
        batch_size=batch_size,
        mixed_precision=mixed_precision,
        devices=devices)

    timer = utils.Timer()
    with tqdm(desc='train', total=num_batches, position=0) as train_pbar:
        train_iter = utils.Generator(trainer.train_iter(model, datasets['labeled']))
        eval_acc = None

        for i, stats in enumerate(train_iter):
            train_pbar.set_postfix(loss=stats.loss, eval_acc=eval_acc, refresh=False)
            train_pbar.update()
            train_logger.write(loss=stats.loss, time=timer())

            if (checkpoint_interval is not None and i > 0 and (i+1) % checkpoint_interval == 0) \
                    or (i == num_batches - 1):
                eval_acc = evaluate(stats.model, stats.avg_model, i+1)
                checkpoint(stats.model, stats.avg_model, stats.optimizer, stats.scheduler, i+1)
                logger.info('eval acc = %.4f' % eval_acc)

            train_logger.step()


if __name__ == '__main__':
    fire.Fire(main)
