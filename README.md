# Sinkhorn Label Allocation: Semi-Supervised Classification via Annealed Self-Training

*ICML 2021*

[[paper]](https://arxiv.org/abs/2102.08622)

<p align="center">
  <img align="middle" src="./assets/schematic.png" alt="Schematic illustration of Sinkhorn Label Allocation" width="80%" />
</p>

## Overview

Semi-supervised learning (SSL) is the setting where the learner is given access to a collection of unlabeled data in
addition to a labeled dataset.
The goal is to learn a more accurate predictor than would otherwise be possible using the labeled data alone. 
Self-training is a standard approach to SSL where the learner's own predictions on unlabeled data are used as 
supervision during training.
As one may expect, the success of self-training depends crucially on the label assignment step: if too many unlabeled
examples are incorrectly labeled, we may end up in a situation where prediction errors compound over the course of
training, ultimately resulting in a poor predictor.
Consequently, practitioners have developed a wide range of label assignment heuristics which serve to mitigate the label
noise introduced through the self-training process.
For example, a commonly seen heuristic is to assign a label only if the current predictor's confidence exceeds a certain
threshold.

In [our paper](https://arxiv.org/abs/2102.08622), we reframe the label assignment process in self-training as an 
optimization problem which aims to find a minimum cost matching between unlabeled examples and classes, subject to a
set of constraints.
As it turns out, this formulation is sufficiently flexible to subsume a variety of popular label assignment heuristics,
e.g., confidence thresholding, label annealing, class balancing, and others.
At the same time, the particular form of the optimization problem admits an efficient approximation algorithm -- the 
Sinkhorn-Knopp algorithm -- thus making it possible to run this assignment procedure within the inner loop of standard
stochastic optimization algorithms.
We call the resulting label assignment process Sinkhorn Label Allocation, or SLA for short.
When combined with consistency regularization, SLA yields a self-training algorithm that achieves strong performance on 
semi-supervised versions of CIFAR-10, CIFAR-100 and SVHN.

## Citation

If you've found this repository useful in your own work, please consider citing our ICML paper:

```
@inproceedings{tai2021sinkhorn,
  title = {{Sinkhorn Label Allocation: Semi-supervised classification via annealed self-training}},
  author = {Tai, Kai Sheng and Bailis, Peter and Valiant, Gregory},
  booktitle = {International Conference on Machine Learning},
  year = {2021},
}
```

## Environment

We recommend using a `conda` environment to manage dependencies:

```sh
$ conda env create -f environment.yml
$ conda activate sinkhorn-label-allocation
```

## Usage 

SLA can be run with a basic set of options using the following command:

```sh
$ python run_sla.py --dataset cifar10 --data_path /tmp/data --output_dir /tmp/sla --run_id my_sla_run --num_labeled 40 --seed 1 --num_epochs 1024 
```

Similarly, the FixMatch baseline can be run using `run_fixmatch.py`:

```sh
$ python run_fixmatch.py --dataset cifar10 --data_path /tmp/data --output_dir /tmp/sla --run_id my_fixmatch_run --num_labeled 40 --seed 1 --num_epochs 1024 
```

The following datasets are currently supported: `cifar10`, `cifar100`, and `svhn`.

A complete mixed precision SLA training run with the default parameters on CIFAR-10 takes about 35 hours on a single 
NVIDIA Titan V.   

For additional algorithm specific options, use the `--help` flag:

```sh
$ python run_supervised.py -- --help
$ python run_fixmatch.py -- --help
$ python run_sla.py -- --help
```

## License

MIT
