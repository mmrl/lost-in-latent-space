"""
dSprites dataset module

The module contains the code for loading the dSprites dataset. This dataset
contains transformations of simple sprites in 2 dimensions, which have no
detailed features.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""


import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data.dataset import Dataset
from functools import partialmethod


class DSprites(Dataset):
    """
    Disentanglement dataset from Loic et al, (2017).

    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # shape:               1=heart, 2=ellipsis, 3=square                      3
    # scale                uniform in range [0.5, 1.0]                        6
    # orientation          uniform in range [0, 2 * pi]                      40
    # position x           unifrom in range [0, 1]                           36
    # position y           unifrom in range [0, 1]                           36
    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/"
                     "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "../data/raw/dsprites/dsprite_train.npz"}

    n_factors = 5

    factors = ('shape', 'scale', 'orientation', 'posX', 'posY')

    factor_sizes = np.array([3, 6, 40, 32, 32])

    categorical = np.array([1, 0, 0, 0, 0])

    img_size = (1, 64, 64)

    unique_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                       0.12903226, 0.16129032, 0.19354839,
                                       0.22580645, 0.25806452, 0.29032258,
                                       0.32258065, 0.35483871, 0.38709677,
                                       0.41935484, 0.4516129, 0.48387097,
                                       0.51612903, 0.5483871, 0.58064516,
                                       0.61290323, 0.64516129, 0.67741935,
                                       0.70967742, 0.74193548, 0.77419355,
                                       0.80645161, 0.83870968, 0.87096774,
                                       0.90322581, 0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                    0.12903226, 0.16129032, 0.19354839,
                                    0.22580645, 0.25806452, 0.29032258,
                                    0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097,
                                    0.51612903, 0.5483871, 0.58064516,
                                    0.61290323, 0.64516129, 0.67741935,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774,
                                    0.90322581, 0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463,
                                           0.48332195, 0.64442926, 0.80553658,
                                           0.96664389, 1.12775121, 1.28885852,
                                           1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242,
                                           2.41660973, 2.57771705, 2.73882436,
                                           2.89993168, 3.061039, 3.22214631,
                                           3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902,
                                           4.34989752, 4.51100484, 4.67211215,
                                           4.83321947, 4.99432678, 5.1554341,
                                           5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799,
                                           6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  }

    def __init__(self, imgs, factor_values, factor_classes,
                 target_transform=None):
        self.imgs = imgs
        self.factor_values = factor_values
        self.factor_classes = factor_classes

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32)]

        self.transform = trans.Compose(image_transforms)
        self.target_transform = target_transform

    def __getitem__(self, key):
        return (self.transform(self.imgs[key]),
                self.factor_values[key],
                self.factor_classes[key])

    def __len__(self):
        return len(self.imgs)

    def __str__(self) -> str:
        return 'dSprites'


def load_raw(path, factor_filter=None):
    data_zip = np.load(path, allow_pickle=True)

    imgs = data_zip['imgs'] * 255
    factor_values = data_zip['latents_values'][:, 1:]  # Remove luminescence
    factor_classes = data_zip['latents_classes'][:, 1:]

    if factor_filter is not None:
        idx = factor_filter(factor_values, factor_classes)

        imgs = imgs[idx]
        factor_values = factor_values[idx]
        factor_classes = factor_classes[idx]

        if len(imgs) == 0:
            raise ValueError('Incorrect masking removed all data')

    # imgs = torch.from_numpy(imgs).to(dtype=torch.float32)
    # factor_values = torch.from_numpy(factor_values).to(dtype=torch.float32)
    # factor_classes = torch.from_numpy(factor_classes).to(dtype=torch.float32)

    return imgs, factor_values, factor_classes


def load(data_filters=(None, None), train=True, path=None):

    train_filter, test_filter = data_filters

    if path is None:
        path = DSprites.files['train']

    if train:
        data = DSprites(*load_raw(path, train_filter))
    else:
        data = DSprites(*load_raw(path, test_filter))

    return data
