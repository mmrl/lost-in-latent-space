"""
Training supervied disentangled models.
"""


import sys
import numpy as np
import torch
from torch.utils.data import random_split
from sacred import Experiment
from sacred.observers import FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer


# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, get_data_spliters
from ingredients.models import model, init_pred
from ingredients.training import training, ModelCheckpoint, init_loss, \
                                 init_metrics, init_loader, init_optimizer, \
                                 xent_loss, mse_loss, accuracy

import configs.training as train_params
import configs.feedforward as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer


# Set up experiment
ex = Experiment(name='supervised', ingredients=[dataset, model, training])

# Observers
ex.observers.append(FileStorageObserver.create('../data/sims/supervised'))

# General configs
ex.add_config(no_cuda=False, save_folder='../data/temp/supervised')
ex.add_package_dependency('torch', torch.__version__)


# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


# Dataset configs
dataset.add_config(setting='supervised', pred_type='reg', shuffle=True,
                   norm_lats=True)


# Training config
training.config(train_params.prediction)
training.add_named_config('multiclass', loss=xent_loss,
                          metrics=[xent_loss, accuracy])
training.add_named_config('regression', loss=mse_loss, metrics=[mse_loss])


# Model config
model.named_config(model_params.abdi)
model.named_config(model_params.kim)
model.named_config(model_params.montero)

# Print the model
@model.command(unobserved=True)
def show():
    model = init_pred(input_size=(3, 64, 64), n_targets=5)
    print(model)


@ex.automain
def main(_config):
    epochs = _config['training']['epochs']

    device = set_seed_and_device()

    data_filters = get_data_spliters()
    dataset = get_dataset(train=True, data_filters=data_filters)

    train_loader, val_loader = init_loader(dataset)

    model = init_pred(input_size=dataset.img_size,
                      n_targets=dataset.n_targets).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(val_loader)

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.should_terminate = True

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        ex.log_scalar('training_loss', tracer.loss[-1])
        tracer.loss.clear()

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    best_checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='classification',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                                {'model': model})

    # Run the training
    trainer.run(train_loader, max_epochs=epochs)

    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')
