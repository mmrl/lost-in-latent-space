import torch
import numpy as np
from numpy.random import laplace
from torch.utils.data.dataset import Dataset
from torch.nn.functional import one_hot
from .wrappers import ImbalancedSampler


class IndexMap:
    """
    Index map that for a given index in the full dataset, returns the corresponding
    index after applying a filter that excludes some factor combinations.

    Use this to sample relevant combinations for Pairs and Triplet dataset for the
    dataset variants with natural statistics as in Klindt et al or for the composition
    task as in Montero et al.
    """
    def __init__(self, dataset):
        total_combs = np.prod(dataset.factor_sizes)

        self.code_bases = total_combs / np.cumprod(dataset.factor_sizes)

        if total_combs == len(dataset):
            index_table = None
        else:
            index_table = np.zeros(np.prod(dataset.factor_sizes),
                                   dtype=np.int64) - 1

            for i, c in enumerate(dataset.factor_classes):
                index_table[self.index(c)] = i

        self.index_table = index_table

    def __getitem__(self, item):
        if self.index_table is None:
            return item
        return self.index_table[item]

    def index(self, code):
        return np.asarray(np.dot(code, self.code_bases), np.int64)

    def is_valid(self, code):
        return self[self.index(code)] != -1


class TupleDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.index_map = IndexMap(base_dataset)

    def __len__(self):
        return len(self.dataset)
        # return self.n_samples

    @property
    def n_factors(self):
        return len(self.factor_sizes)

    @property
    def factor_sizes(self):
        return self.dataset.factor_sizes

    @property
    def img_size(self):
        return self.dataset.img_size

    @property
    def factors(self):
        return self.dataset.factors

    @property
    def factor_code(self):
        return self.dataset.factor_classes

    @property
    def factor_values(self):
        return self.dataset.factor_values

    @property
    def imgs(self):
        return self.dataset.imgs

    @property
    def transform(self):
        return self.dataset.transform

    @property
    def categorical(self):
        return self.dataset.categorical

    def code2idx(self, code):
        return self.index_map.index(code)

    def code2img(self, code):
        idx = self.index_map[self.code2idx(code)]
        return self.imgs[idx]

    def get_balanced_sampler(self, factor):
        cat_idx = [i for i, s in enumerate(self.factors) if factor in s][0]
        labels = self.factor_values[:,cat_idx]
        return ImbalancedSampler(self, labels)


class PairDataset(TupleDataset):
    def __init__(self, dataset, rate=-1, distribution='lap'):
        super().__init__(dataset)

        assert distribution in ['unif', 'lap']

        self.rate = rate
        self.dist = distribution

    def __getitem__(self, idx):
        img, _, z_code = self.dataset[idx]

        z_transf_code = self.sample_factor(z_code)
        img_transf = self.imgs[self.code2idx(z_transf_code)]

        if self.transform:
            img = self.transform(img)
            img_transf = self.transform(img_transf)

        inputs = torch.stack([img, img_transf], dim=0).contiguous()

        return inputs, inputs

    def sample_factor(self, z_code):
        if self.rate == -1:
            rate = np.random.uniform(1, 10, 1)
        else:
            rate = self.rate

        transf_code = z_code.copy()

        factor_idx = np.arange(self.n_factors)
        np.random.shuffle(factor_idx)

        one_has_changed = False  # To ensure at leat one dimension changes
        for i in factor_idx:
            if self.categorical[i]:
                continue

            mean, n_values = transf_code[i], self.factor_sizes[i]

            # Compute probabilities
            possible_values = np.arange(n_values)

            if self.dist == 'lap':
                p = laplace.pdf(possible_values, loc=mean,
                                scale=np.log(n_values) / rate)
            else:
                p = np.ones(n_values)

            # Set invalid combinations to 0
            possible_codes = np.repeat(z_code[None], [n_values], axis=0)
            is_valid = self.index_map.is_valid(possible_codes)
            p[~is_valid] = 0

            # Ensure at leat one dimension changes
            if not one_has_changed and sum(is_valid) > 1:
               p[mean] = 0
               one_has_changed = True

            p /= p.sum()
            transf_code[i] = np.random.choice(possible_values, p=p)

        return transf_code


class TripletDataset(TupleDataset):
    def __getitem__(self, idx):
        z_code = self.factor_code[idx]
        transf_z_code = z_code.copy()

        # select factor to transform and value to transform to
        all_factors = np.arange(self.n_factors)
        np.random.shuffle(all_factors)

        # Iterate through all dimensions until we sample a new value
        dim = None
        for dim in all_factors:
            new_dim_code = self.sample_factor(z_code, dim)

            # Only assign if we could sample a new value for that dimension
            # if it's the last dimension, then we don't have any other choice
            if (new_dim_code != transf_z_code[dim]) or (dim == all_factors[-1]):
                transf_z_code[dim] = new_dim_code
                break

        # sample a command image
        command_z_code = transf_z_code.copy()
        for d in range(len(self.factor_sizes)):
            if d != dim:
                command_z_code[d] = self.sample_factor(command_z_code, d)

        action = one_hot(torch.LongTensor([dim]),
                         num_classes=self.n_factors).squeeze()

        img = self.transform(self.imgs[idx])
        command_img = self.transform(self.code2img(command_z_code))
        transformed_img = self.transform(self.code2img(transf_z_code))

        input_imgs = torch.stack([img, command_img], dim=0).contiguous()
        target = torch.stack([img, command_img, transformed_img],
                             dim=0).contiguous()

        return (input_imgs, action), target

    def sample_factor(self, factor_code, dim):
        factor_d_code = np.arange(self.factor_sizes[dim])

        # Determine which codes are valid
        possible_codes = np.repeat(factor_code[None],
                                   [len(factor_d_code)], axis=0)
        possible_codes[:, dim] = factor_d_code

        idxs = self.code2idx(possible_codes)
        is_valid = self.index_map[idxs] != -1

        # If more than one value is valid, remove the current one
        if sum(is_valid) > 1:
            is_valid[factor_code[dim]] = False
        # else return current
        else:
            return factor_code[dim]

        prob = np.ones(self.factor_sizes[dim]) / (sum(is_valid))
        prob[~is_valid] = 0

        return np.random.choice(factor_d_code, p=prob)
