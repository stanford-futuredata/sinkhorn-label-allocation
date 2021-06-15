import numpy as np
from torchvision import datasets, transforms
import logging
from augmentations import RandAugment, Maybe, cutout_tensor
from functools import partial
from PIL import Image


logger = logging.getLogger(__name__)

default_mean = (0., 0., 0.)
default_std = (1., 1., 1.)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)


def get_cifar10(
        data_root,
        num_labeled,
        labeled_aug='weak',
        unlabeled_aug='strong',
        sample_mode='label_dist',
        whiten=True,
        incl_labeled_in_unlabeled=True):
    base_dataset = datasets.CIFAR10(data_root, train=True, download=True)
    if num_labeled is None:
        num_labeled = len(base_dataset)

    if whiten:
        mean, std = cifar10_mean, cifar10_std
    else:
        mean, std = default_mean, default_std

    # transformations
    transform_labeled = get_transform(mean, std, mode=labeled_aug)
    transform_unlabeled = get_transform(mean, std, mode=unlabeled_aug)
    transform_val = get_transform(mean, std, mode='none')

    # split indices
    labeled_idxs, unlabeled_idxs = labeled_unlabeled_split(
        base_dataset.targets, num_labeled,
        sample_mode=sample_mode,
        incl_labeled_in_unlabeled=incl_labeled_in_unlabeled)

    # split datasets
    labeled_dataset = CIFAR10(data_root, labeled_idxs, train=True, transform=transform_labeled)
    unlabeled_dataset = CIFAR10(data_root, unlabeled_idxs, train=True, transform=transform_unlabeled, target_idx=True)
    test_dataset = CIFAR10(data_root, train=False, transform=transform_val, download=True)

    logger.info("dataset: CIFAR10")
    logger.info(f"labeled examples: {len(labeled_idxs)}")
    logger.info(f"unlabeled examples: {len(unlabeled_idxs)}")

    return dict(base=base_dataset, labeled=labeled_dataset, unlabeled=unlabeled_dataset, test=test_dataset)


def get_cifar100(
        data_root,
        num_labeled,
        labeled_aug='weak',
        unlabeled_aug='strong',
        sample_mode='label_dist',
        whiten=True,
        incl_labeled_in_unlabeled=True):
    base_dataset = datasets.CIFAR100(data_root, train=True, download=True)
    if num_labeled is None:
        num_labeled = len(base_dataset)

    if whiten:
        mean, std = cifar100_mean, cifar100_std
    else:
        mean, std = default_mean, default_std

    transform_labeled = get_transform(mean, std, mode=labeled_aug)
    transform_unlabeled = get_transform(mean, std, mode=unlabeled_aug)
    transform_val = get_transform(mean, std, mode='none')

    # split indices
    labeled_idxs, unlabeled_idxs = labeled_unlabeled_split(
        base_dataset.targets, num_labeled,
        sample_mode=sample_mode,
        incl_labeled_in_unlabeled=incl_labeled_in_unlabeled)

    # split datasets
    labeled_dataset = CIFAR100(data_root, labeled_idxs, train=True, transform=transform_labeled)
    unlabeled_dataset = CIFAR100(data_root, unlabeled_idxs, train=True, transform=transform_unlabeled, target_idx=True)
    test_dataset = CIFAR100(data_root, train=False, transform=transform_val, download=True)

    logger.info("dataset: CIFAR100")
    logger.info(f"labeled examples: {len(labeled_idxs)}")
    logger.info(f"unlabeled examples: {len(unlabeled_idxs)}")

    return dict(base=base_dataset, labeled=labeled_dataset, unlabeled=unlabeled_dataset, test=test_dataset)


def get_svhn(
        data_root,
        num_labeled,
        labeled_aug='weak_noflip',
        unlabeled_aug='strong_noflip',
        whiten=True,
        sample_mode='label_dist',
        incl_labeled_in_unlabeled=True):
    base_dataset = datasets.SVHN(data_root, split='train', download=True)
    if num_labeled is None:
        num_labeled = len(base_dataset)

    if whiten:
        mean, std = svhn_mean, svhn_std
    else:
        mean, std = default_mean, default_std

    transform_labeled = get_transform(mean, std, mode=labeled_aug)
    transform_unlabeled = get_transform(mean, std, mode=unlabeled_aug)
    transform_val = get_transform(mean, std, mode='none')

    # split indices
    labeled_idxs, unlabeled_idxs = labeled_unlabeled_split(
        base_dataset.labels, num_labeled,
        sample_mode=sample_mode,
        incl_labeled_in_unlabeled=incl_labeled_in_unlabeled)

    # split datasets
    labeled_dataset = SVHN(data_root, labeled_idxs, train=True, transform=transform_labeled)
    unlabeled_dataset = SVHN(data_root, unlabeled_idxs, train=True, transform=transform_unlabeled, target_idx=True)
    test_dataset = SVHN(data_root, train=False, transform=transform_val, download=True)

    logger.info("dataset: SVHN")
    logger.info(f"labeled examples: {len(labeled_idxs)}")
    logger.info(f"unlabeled examples: {len(unlabeled_idxs)}")

    return dict(base=base_dataset, labeled=labeled_dataset, unlabeled=unlabeled_dataset, test=test_dataset)


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, idxs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, target_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.target_idx = target_idx
        self.idxs = idxs
        if idxs is not None:
            self.data = self.data[idxs]
            if self.target_idx:
                self.targets = idxs
            else:
                self.targets = np.array(self.targets)[idxs]

    @property
    def num_classes(self):
        return 10

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, idxs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, target_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.target_idx = target_idx
        self.idxs = idxs
        if idxs is not None:
            self.data = self.data[idxs]
            if self.target_idx:
                self.targets = idxs
            else:
                self.targets = np.array(self.targets)[idxs]

    @property
    def num_classes(self):
        return 100

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN(datasets.SVHN):
    def __init__(self, root, idxs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, target_idx=False):
        super().__init__(root, split='train' if train else 'test',
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.target_idx = target_idx
        self.idxs = idxs
        self.targets = self.labels  # SVHN base class uses `labels` attribute unlike CIFAR10/100
        self.data = self.data.transpose([0, 2, 3, 1])
        if idxs is not None:
            self.data = self.data[idxs]
            if self.target_idx:
                self.targets = idxs
            else:
                self.targets = np.array(self.targets)[idxs]

    @property
    def num_classes(self):
        return 10

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def labeled_unlabeled_split(
        labels,
        num_labeled,
        sample_mode,
        incl_labeled_in_unlabeled):
    labels = np.array(labels)

    # compute class distribution
    classes, class_counts = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    class_dist = class_counts / class_counts.sum()

    # allocate labels to each class
    if sample_mode == 'equal':
        if num_labeled % num_classes != 0:
            raise ValueError('Number of labels must be divisible by number of classes for equal label allocation')
        labels_per_class = np.full(num_classes, num_labeled // num_classes)
    elif sample_mode == 'multinomial':
        labels_per_class = np.random.multinomial(num_labeled, class_dist)
    elif sample_mode == 'multinomial_min1':
        if num_labeled < num_classes:
            raise ValueError('Number of labels must be at least the number of classes')
        labels_per_class = np.random.multinomial(num_labeled - num_classes, class_dist) + 1
    elif sample_mode == 'label_dist':
        labels_per_class = (class_dist * num_labeled).astype(int)
        for _ in range(num_labeled - labels_per_class.sum()):
            i = np.argmax(class_dist - labels_per_class / labels_per_class.sum())
            labels_per_class[i] += 1
    elif sample_mode == 'label_dist_min1':
        if num_labeled < num_classes:
            raise ValueError('Number of labels must be at least the number of classes')
        labels_per_class = (class_dist * num_labeled).astype(int)
        for i in range(num_classes):
            if labels_per_class[i] == 0:
                labels_per_class[i] += 1
        for _ in range(num_labeled - labels_per_class.sum()):
            i = np.argmax(class_dist - labels_per_class / labels_per_class.sum())
            labels_per_class[i] += 1
    else:
        raise ValueError('Invalid sampling mode {}'.format(sample_mode))

    # randomly select examples according to label allocation
    labeled_idxs = []
    unlabeled_idxs = []
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        labeled_idxs.extend(idxs[:labels_per_class[i]])
        if incl_labeled_in_unlabeled:
            unlabeled_idxs.extend(idxs)
        else:
            unlabeled_idxs.extend(idxs[labels_per_class[i]:])

    return np.array(labeled_idxs), np.array(unlabeled_idxs)


def get_transform(mean, std, mode):
    if type(mode) is tuple:
        if len(mode) == 1:
            mode = mode[0]
        return TransformTuple(mean, std, mode)

    if mode == 'none':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if mode == 'weak':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if mode == 'weak_noflip':
        return transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if mode == 'strong':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugment(num_ops=2, num_levels=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            partial(cutout_tensor, size=16),
        ])

    if mode == 'strong_noflip':
        return transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugment(num_ops=2, num_levels=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            partial(cutout_tensor, size=16),
        ])

    raise ValueError("Invalid mode: {}".format(mode))


class TransformTuple(object):
    def __init__(self, mean, std, modes):
        self.transforms = [get_transform(mean, std, mode) for mode in modes]

    def __call__(self, x):
        return tuple(tf(x) for tf in self.transforms)
