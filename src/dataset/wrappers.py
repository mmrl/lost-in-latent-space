"""
Wrappers for the different kinds of training settings we want to use
"""

import numpy as np
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class Wrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    @property
    def n_factors(self):
        return self.base_dataset.n_factors

    @property
    def factors(self):
        return self.base_dataset.factors

    @property
    def imgs(self):
        return self.base_dataset.imgs

    @property
    def factor_values(self):
        return self.base_dataset.factor_values

    @property
    def factor_classes(self):
        return self.base_dataset.factor_classes

    @property
    def img_size(self):
        return self.base_dataset.img_size

    @property
    def factor_sizes(self):
        return self.base_dataset.factor_sizes

    @property
    def unique_values(self):
        return self.base_dataset.unique_values

    @property
    def transform(self):
        return self.base_dataset.transform

    @property
    def target_transform(self):
        return self.base_dataset.target_transform

    def get_balanced_sampler(self, factor):
        cat_idx = [i for i, s in enumerate(self.factors) if factor in s][0]
        labels = self.factor_values[:,cat_idx]
        return ImbalancedSampler(self, labels)


class Supervised(Wrapper):
    def __init__(self, base_dataset, dim=None,
                 pred_type='reg', norm_lats=True):
        super().__init__(base_dataset)
        self.pred_type = pred_type
        self.dim = dim
        self.norm_lats = norm_lats

        if norm_lats:
            mean_values, min_values, max_values = [], [], []
            for f in base_dataset.factors:
                mean_values.append(base_dataset.unique_values[f].mean())
                min_values.append(base_dataset.unique_values[f].min())
                max_values.append(base_dataset.unique_values[f].max())

            self._mean_values = np.array(mean_values)
            self._min_values = np.array(min_values)
            self._max_values = np.array(max_values)

            def standarize(factor_values):
                return ((factor_values - self._mean_values) /
                         (self._max_values - self._min_values))

            self.standarize = standarize

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])

        if self.pred_type == 'class':
            target = self.factor_classes[idx]
        else:
            target = self.factor_values[idx]

        if self.target_transform:
            target = self.target_transform(target)

        if self.norm_lats:
            target = self.standarize(target).astype(np.float32)

        if self.dim is not None:
            target = target[self.dim]

        return img, target

    def __str__(self):
        return 'Supervised{}'.format(str(self.base_dataset))

    @property
    def n_targets(self):
        if self.pred_type == 'class':
            if self.dim is not None:
                return self.factor_sizes[self.dim]
            else:
                return self.factor_sizes
        if self.dim is not None:
            return 1
        return self.n_factors


class Unsupervised(Wrapper):
    def __getitem__(self, idx):
        img = self.imgs[idx]

        if self.target_transform:
            img = self.transform(img)
            target = self.target_transform(img)
            return img, target

        img = self.transform(img)

        return img, img

    def __str__(self):
        return 'Unsupervised{}'.format(str(self.base_dataset))


class Reconstruction(Wrapper):
    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])
        target = self.factor_values[idx]

        if self.target_transform:
            target = self.target_transform(target)

        return target, img

    def __str__(self):
        return 'Reconstruction{}'.format(str(self.base_dataset))


class ImbalancedSampler(Sampler):
    """
    Rebalances a dataset so that all labels are presented the same amount of times.

    Based on code found (here)[https://github.com/ufoym/imbalanced-dataset-sampler].
    """
    def __init__(self, dataset, labels) -> None:
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] =  labels
        df.index = pd.Index(self.indices)
        df.sort_index(inplace=True)

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights,
            self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
