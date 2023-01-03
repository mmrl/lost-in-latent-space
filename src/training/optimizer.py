"""
Helper functions to initialize an optimizaton algorithm.
"""


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def init_optimizer(optimizer, params, lr=0.01, l2_norm=0.0, **kwargs):

    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=l2_norm, **kwargs)
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params, lr=lr, **kwargs)
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params, lr=lr, weight_decay=l2_norm, **kwargs)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr,
                                  weight_decay=l2_norm, **kwargs)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr,
                              weight_decay=l2_norm, **kwargs)  # 0.01
    elif optimizer == 'nesterov':
        optimizer = optim.SGD(params, lr=lr, weight_decay=l2_norm,
                              nesterov=True, **kwargs)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=lr,
                                  weight_decay=l2_norm, **kwargs)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params, lr=lr,
                                   weight_decay=l2_norm, **kwargs)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer
