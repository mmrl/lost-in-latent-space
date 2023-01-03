"""
3DShapes dataset module

The module contains the code for loading the 3DShapes dataset. The dataset can
be loaded in 3 modes: supervised, unsupervised, and ground-truth factor
reconstruction. We mostly use the last for 3 training and the first one for
analyzing the results. Data loading of the batches is handled in the
corresponding Sacred ingredient.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""
from itertools import product

import numpy as np
import h5py
import torch
import torchvision.transforms as trans
from skimage.color import rgb2hsv
from torch.utils.data import Dataset
# from functools import partialmethod


class Shapes3D(Dataset):
    """
    Disentangled dataset used in Kim and Mnih, (2019)

    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # floor hue:           uniform in range [0.0, 1.0)                      10
    # wall hue:            uniform in range [0.0, 1.0)                      10
    # object hue:          uniform in range [0.0, 1.0)                      10
    # scale:               uniform in range [0.75, 1.25]                     8
    # shape:               0=square, 1=cylinder, 2=sphere, 3=pill            4
    # orientation          uniform in range [-30, 30]                       15
    """
    files = {"train": "../data/raw/shapes3d/3dshapes.h5"}

    n_factors = 6

    factors = ('floor_hue', 'wall_hue', 'object_hue',
               'scale', 'shape', 'orientation')

    factor_sizes = np.array([10, 10, 10, 8, 4, 15])

    categorical = np.array([0, 0, 0, 0, 1, 0])

    img_size = (3, 64, 64)

    unique_values = {'floor_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                            0.5, 0.6, 0.7, 0.8, 0.9]),
                     'wall_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                           0.5, 0.6, 0.7, 0.8, 0.9]),
                     'object_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                             0.5, 0.6, 0.7, 0.8, 0.9]),
                     'scale': np.array([0.75, 0.82142857, 0.89285714, 0.96428571,
                               1.03571429, 1.10714286, 1.17857143, 1.25]),
                     'shape': np.array([0, 1, 2, 3]),
                     'orientation': np.array([-30., -25.71428571, -21.42857143,
                                     -17.14285714, -12.85714286, -8.57142857,
                                     -4.28571429, 0., 4.28571429, 8.57142857,
                                     12.85714286, 17.14285714, 21.42857143,
                                     25.71428571,  30.])}

    def __init__(self, imgs, factor_values, factor_classes, color_mode='rgb',
                 target_transform=None):
        self.imgs = imgs
        self.factor_values = factor_values
        self.factor_classes = factor_classes

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32)]

        if color_mode == 'hsv':
            image_transforms.insert(0, trans.Lambda(rgb2hsv))

        self.transform = trans.Compose(image_transforms)
        self.target_transform = target_transform

    def __getitem__(self, key):
        return (self.transform(self.imgs[key]),
                self.factor_values[key],
                self.factor_classes[key])

    def __len__(self):
        return len(self.imgs)

    def __str__(self) -> str:
        return '3DShapes'


def load_raw(path, factor_filter=None):
    data_zip = h5py.File(path, 'r')

    imgs = data_zip['images'][()]
    factor_values = data_zip['labels'][()]
    factor_classes = np.asarray(list(product(
        *[range(i) for i in Shapes3D.factor_sizes])))

    if factor_filter is not None:
        idx = factor_filter(factor_values, factor_classes)

        imgs = imgs[idx]
        factor_values = factor_values[idx]
        factor_classes = factor_classes[idx]

        if len(imgs) == 0:
            raise ValueError('Incorrect masking removed all data')

    return imgs, factor_values, factor_classes


def load(data_filters=(None, None), train=True, color_mode='rgb', path=None):

    train_filter, test_filter = data_filters

    if path is None:
        path = Shapes3D.files['train']

    if train:
        data = Shapes3D(*load_raw(path, train_filter), color_mode=color_mode)
    else:
        data = Shapes3D(*load_raw(path, test_filter), color_mode=color_mode)

    return data
