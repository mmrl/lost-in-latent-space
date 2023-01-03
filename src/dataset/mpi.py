"""
Disentanglement dataset from in Gondal et al 2019.

This dataset contains more realistic stimuli when compared to dSprites and
3Dshapes. Plus some combinations of factors have a tighther coupling between
them than others, which means that the models have an harder/easier time
learning how they interact.

For more info, the dataset can be found here:

arXiv preprint https://arxiv.org/abs/1906.03292
NeurIPS Challenge: https://www.aicrowd.com/challenges/
                           neurips-2019-disentanglement-challenge
"""


import numpy as np
import torch
import torchvision.transforms as trans
from itertools import product
from skimage.color import rgb2hsv
from torch.utils.data.dataset import Dataset
# from functools import partialmethod


class MPI3D(Dataset):
    """
    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # object color:        white=0, green=1, red=2, blue=3,                  6
    #                      brown=4, olive=5
    # object shape:        cone=0, cube=1, cylinder=2,                       6
    #                      hexagonal=3, pyramid=4, sphere=5
    # object size:         small=0, large=1                                  2
    # camera height:       top=0, center=1, bottom=2                         3
    # background color:    purple=0, sea green=1, salmon=2                   3
    # horizontal axis:     40 values liearly spaced [0, 39]                 40
    # vertical axis:       40 values liearly spaced [0, 39]                 40
    """
    files = {"toy": "../data/raw/mpi3d/mpi3d_toy.npz",
             "realistic": "../data/raw/mpi3d/mpi3d_realistic.npz",
             "real": "../data/raw/mpi3d/mpi3d_real.npz"}

    n_factors = 7
    factors = ('object_color', 'object_shape', 'object_size', 'camera_height',
               'background_color', 'horizontal_axis', 'vertical_axis')
    factor_sizes = np.array([6, 6, 2, 3, 3, 40, 40])

    categorical = np.array([0, 1, 1, 1, 1, 0, 0])

    img_size = (3, 64, 64)

    unique_values = { 'object_color'    : np.arange(6),
                      'object_shape'    : np.arange(6),
                      'object_size'     : np.arange(2),
                      'camera_height'   : np.arange(3),
                      'background_color': np.arange(3),
                      'horizontal_axis' : np.arange(40),
                      'vertical_axis'   : np.arange(40)}

    def __init__(self, imgs, factor_values, factor_classes, version='real',
                 target_transform=None,
                 color_mode='rgb'):
        self.imgs = imgs
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.version = version

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
        return 'MPI3D<{}>'.format(self.version)


def load_raw(path, factor_filter=None):
    data_zip = np.load(path, allow_pickle=True)
    images = data_zip['images']

    factor_values = list(product(*MPI3D.unique_values.values()))
    factor_values = np.asarray(factor_values, dtype=np.int8)
    factor_classes = np.asarray(list(product(
       *[range(i) for i in MPI3D.factor_sizes])))

    if factor_filter is not None:
        idx = factor_filter(factor_values,factor_classes)

        images = images[idx]
        factor_values = factor_values[idx]

        if len(images) == 0:
            raise ValueError('Incorrect masking removed all data')

    return images, factor_values.astype(np.float32), factor_values


def load(version='real', data_filters=(None, None), color_mode='rgb',
         train=True, path=None):

    if version not in ['toy', 'realistic', 'real']:
        raise ValueError('Unrecognized datset version {}'.format(version))

    if path is None:
        path = MPI3D.files[version]

    if train:
        data = MPI3D(*load_raw(path, data_filters[0]), version=version,
                     color_mode=color_mode)
    else:
        data = MPI3D(*load_raw(path, data_filters[1]), version=version,
                     color_mode=color_mode)

    return data
