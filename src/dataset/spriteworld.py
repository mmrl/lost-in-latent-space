import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data.dataset import Dataset
from functools import partial
from itertools import product


class Sprites(Dataset):
    """
    Disentanglement dataset generated from an adapted implementation of the
    Spriteworld framework for reinforcement learning Watters et al, (2019).

    Because these can be generated with different parameters, we do not know
    what their values are before loading one. See the raw/sprites folder
    to access the pickled objects.
    """
    versions = {
            "circles"  : "../data/raw/spriteworld/circles.npz",
            "simple"   : "../data/raw/spriteworld/simple.npz",
            "twoshapes": "../data/raw/spriteworld/twoshapes_ext.npz"}

    def __init__(self, imgs, factor_values, factor_classes,
                 target_transform=None):
        self.imgs = imgs
        self.factor_values = factor_values
        self.factor_classes = factor_classes

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32),
                            trans.Lambda(lambda x: x.flatten())]

        self.transform = trans.Compose(image_transforms)
        self.target_transform = target_transform

    def __getitem__(self, key):
        return (self.transform(self.imgs[key]),
                self.factor_values[key],
                self.factor_classes[key])

    def __len__(self):
        return len(self.imgs)

    def __str__(self) -> str:
        return 'Spriteworld'


def load_raw(path, factor_filter=None):
    data_zip = np.load(path, allow_pickle=True)
    set_meta = partial(setattr, Sprites)

    # Meta values
    name = data_zip['name']
    n_factors = data_zip['n_factors']
    factors = list(data_zip['factors'])
    factor_sizes = tuple(data_zip['factor_sizes'])
    unique_values = data_zip['unique_values'].item()
    img_size = tuple(data_zip['img_size'])

    set_meta('name', name)
    set_meta('n_factors', n_factors)
    set_meta('factors', factors)
    set_meta('factor_sizes', factor_sizes)
    set_meta('unique_values', dict(unique_values))
    set_meta('img_size', img_size)

    # Instance values
    imgs = data_zip['images']
    factor_values = data_zip['factor_values']
    factor_classes = np.asarray(list(product(
       *[range(i) for i in Sprites.factor_sizes])))

    if factor_filter is not None:
        idx = factor_filter(factor_values, factor_classes)

        imgs = imgs[idx]
        factor_values = factor_values[idx]
        factor_classes = factor_classes[idx]

        if len(imgs) == 0:
            raise ValueError('Incorrect masking removed all data')

    # assert len(imgs) == len(factor_classes)

    return imgs, factor_values, factor_classes


def load(data_filters=(None, None), train=True, version=None):

    train_filter, test_filter = data_filters

    # path = Sprites.versions.get(version, Sprites.versions['simple'])
    path = Sprites.versions[version]

    if train:
        data = Sprites(*load_raw(path, train_filter))
    else:
        data = Sprites(*load_raw(path, test_filter))

    return data
