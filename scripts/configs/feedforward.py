"""
Regressor model definitions

We define the architectures of the models here. The names referece the first
author of the article from where they were take. Some might be slightly
modified.

Configurations consist of a list of layers. Input size is determined by the
dataset it will be used on. Parameters in the config for each layer follow
the order in Pytorch's documentation. Excluding any of them will use the
default ones. We can also pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==================================================================================
Convolution:            n-channels, size, stride, padding
Transposed Convolution: same, remeber output_padding when stride > 1! (use kwargs)
Pooling:                size, stride, padding, type
Linear:                 output size, fit bias
Flatten:                start dim, (optional, defaults=-1) end dim
Unflatten:              unflatten shape (have to pass the full shape)
Batch-norm:             dimensionality (1-2-3d)
Upsample:               upsample_shape (hard to infer automatically). Only bilinear
Non-linearity:          pass whatever arguments that non-linearity supports.
"""

def kim():
    layers = [
            ('conv', (32, 4, 2, 1)),
            ('relu',),

            ('conv', (32, 4, 2, 1)),
            ('relu',),

            ('conv', (64, 4, 2, 1)),
            ('relu',),

            ('conv', (64, 4, 2, 1)),
            ('relu',),

            ('flatten', [1]),

            ('linear', [256]),
            ('relu',)
        ]


def abdi():
    layers = [
            ('conv', (32, 4, 2, 1)),
            ('relu',),

            ('conv', (64, 4, 2, 1)),
            ('relu',),

            ('conv', (128, 4, 2, 1)),
            ('relu',),

            ('conv', (256, 4, 2, 1)),
            ('relu',),

            ('flatten', [1]),

            ('linear', [256]),
            ('relu',)
        ]

def montero():
    layers = [
        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (128, 4, 2, 1)),
        ('relu',),

        ('conv', (128, 4, 2, 1)),
        ('relu',),

        ('conv', (256, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]
