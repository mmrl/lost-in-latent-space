"""
Training configurations for disentangled and ground-truth decoder models

Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names reference the loss functions as
defined in the corresponding articles (see src/training/loss.py). Both use the
following format:

    {'name': <name>, 'params': <dict_of_kwargs>, 'label': <alt_name>}

The parameters are the ones specified by each fuction (check the file above).
Alt name is used to resolve conflcits between metrics with the same parameters^*.


^* This is addmitedly a hack I had to use due to the way Ignite works.
"""


################################## Disentangled models ########################

def vae():
    """
    Default training parameters
    """
    epochs              = 66
    batch_size          = 64
    optimizer           = 'adam'
    lr                  = 0.0005
    # early_stopping    = False
    # decay_lr          = False
    # lr_scale          = 0.1
    # lr_decay_patience = 10

    loss                = {'name'  : 'beta-vae',
                           'params': {'reconstruction_loss': 'bce'}}
    metrics             = [{'name' : 'recons_nll', 'params': {'loss': 'bce'}}]
               # {'name': 'kl-div', 'params': {}},
               # {'name': 'kl-div', 'params': {'dim': 0}, 'label': 'kl-latent-0'},
               # {'name': 'kl-div', 'params': {'dim': 1}, 'label': 'kl-latent-1'},
               # {'name': 'kl-div', 'params': {'dim': 2}, 'label': 'kl-latent-2'},
               # {'name': 'kl-div', 'params': {'dim': 3}, 'label': 'kl-latent-3'},
               # {'name': 'kl-div', 'params': {'dim': 4}, 'label': 'kl-latent-4'},
               # {'name': 'kl-div', 'params': {'dim': 5}, 'label': 'kl-latent-5'},
               # {'name': 'kl-div', 'params': {'dim': 6}, 'label': 'kl-latent-6'},
               # {'name': 'kl-div', 'params': {'dim': 7}, 'label': 'kl-latent-7'},
               # {'name': 'kl-div', 'params': {'dim': 8}, 'label': 'kl-latent-8'},
               # {'name': 'kl-div', 'params': {'dim': 9}, 'label': 'kl-latent-9'}]


def beta():
    """
    Training parameters as in Higgins et al., 2017
    """
    loss = {'name'  : 'beta-vae',
            'params': {'beta': 4.0,
                       'reconstruction_loss': 'bce'}}


def cci():
    """
    Training parameters as in Burgess et al., 2018
    """
    loss = {'name'  : 'constrained-beta-vae',
            'params': {'reconstruction_loss': 'bce',
                       'gamma': 50.0,
                       'capacity_schedule': [0.0, 20, 10000]}}


def factor():
    """
    Training as in Kim & Mnih 2019
    """
    # lr       = 0.001
    batch_size = 128
    loss       = {'name'  : 'factor-vae',
                  'params': {'gamma': 100,
                            'reconstruction_loss': 'bce'}}
    metrics    = [{'name' : 'recons_nll', 'params': {'loss': 'bce'}},
                  {'name' : 'kl-div', 'params': {}}]


def waegan():
    lr      = 3e-4
    loss    = {'name' : 'wae-gan',
              'params': {'lambda1': 10.0,
                         'prior_var': 2.0,
                         'reconstruction_loss': 'mse'}}
    metrics = [{'name': 'recons_nll', 'params': {'loss': 'mse'}}]


def waemmd():
    lr      = 1e-3
    loss    = {'name': 'wae-mmd',
               'params': {'lambda1': 10.0,
                          'lambda2': 0.001,
                          'prior_var': 2.0,
                          'reconstruction_loss': 'mse'}}

    metrics = [{'name': 'recons_nll', 'params': {'loss': 'mse'}}]


def cascade():
    epochs = 30
    lr     = 3e-4
    loss   = {'name': 'infocasc',
              'params': {'n_cont': 6,
                         'beta': (2.0, 10.0),
                         'warmup': 20000}}

def lievae():
    epochs = 50
    batch_size = 256
    lr = 1e-4

    loss = {'name'  : 'lie-vae',
            'params': {'reconstruction_loss': 'mse',
                       'subspace_sizes': [10],
                       'subgroup_sizes': [100],
                       'hy_hes': 40,
                       'hy_rec': 0.1,
                       'hy_commute': 0,
                       'gamma': 1.0,
                       'capacity_schedule': None}}

    metrics = [{'name': 'recons_nll', 'params': {'loss': 'mse'}}]

def banneal():
    loss = {'name'  : 'beta-vae',
            'params': {'reconstruction_loss': 'bce',
                       'beta': 10,
                       'beta_schedule': [1000000, 'anneal', 0.1]}}

def bsched():
    loss = {'name'  : 'beta-vae',
            'params': {'reconstruction_loss': 'bce',
                       'beta': 1,
                       'beta_schedule': [100000, 'increase', 0.0]}}


################################## Decoders ###################################


def generation():
    """
    Training for ground-truth decoders
    """

    epochs              = 500
    batch_size          = 64
    optimizer           = 'adam'
    lr                  = 0.001
    l2_norm             = 0.0
    rate_reg            = 0.0
    clip                = 0.0
    early_stopping      = False
    decay_lr            = False
    lr_scale            = 0.1
    lr_decay_patience   = 10


################################### Predictors ################################

def prediction():
    """
    Training settings for classifiers
    """

    epochs              = 100
    batch_size          = 64
    optimizer           = 'adam'
    lr                  = 0.001
    train_val_split     = 0.2

################################## Schedulers #################################

def step_lr():
    """
    Exponential step decay
    """
    step_size = 50000
    gamma     = 0.3
