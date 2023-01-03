"""
Experiment script to train models on the composition task.

In this task models are with two images and a query vector. The query vector
specifies along which dimension the frist image must be transformed so as to
match the second image along this dimension.

The architectures are based on the normal autoencoder with an additional
``mixer'' netwotrk that transforms the latent representations.

To run:
    cd <experiment-root>/scripts/
    python -m experiment.composition with dataset.<option> model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred
documentation [https://sacred.readthedocs.io/en/stable/]
"""

import sys
import numpy as np
import torch
from functools import partial

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer


# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, init_loader, \
                                get_data_spliters
from ingredients.models import model, init_compnet
from ingredients.training import training, ModelCheckpoint, init_metrics, \
                                 init_optimizer, init_loss

import configs.training as train_params
import configs.vaes as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer

# Set up experiment
ex = Experiment(name='composition', ingredients=[dataset, model, training])

# Observers
ex.observers.append(FileStorageObserver.create('../data/sims/composition'))

# General configs
ex.add_config(no_cuda=False, save_folder='../data/temp/composition')
ex.add_package_dependency('torch', torch.__version__)

# Required for ProgressBar to work properly
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

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


def loss_string(name, output):
    return {f"{name.upper()}": f"{output:,.2f}"}


def metrics_string(metrics):
    output = ['Validation:']
    for k, v in metrics.items():
        output.append(f"{k.upper()}={v:,.2f}")
    return ' '.join(output)


# Dataset configs
dataset.add_config(setting='composition', shuffle=True)


# Training configs
training.config(train_params.vae)
training.named_config(train_params.beta)
training.named_config(train_params.cci)
training.named_config(train_params.factor)
training.named_config(train_params.bsched)
training.named_config(train_params.banneal)
training.named_config(train_params.waemmd)


# Model configs
model.named_config(model_params.higgins)
model.named_config(model_params.burgess_v2)
model.named_config(model_params.kim)
model.named_config(model_params.abdi)
model.named_config(model_params.montero)
model.named_config(model_params.watters)
model.named_config(model_params.sbd3)  # SBD architecture for Circles/Simple
model.named_config(model_params.sbd4)  # SBD architecture for other datasets

# Print the model
@model.command(unobserved=True)
def show():
    model = init_compnet(input_size=(3, 64, 64), n_actions=5)
    print(model)


# Run experiment
@ex.automain
def main(_config):
    epochs = _config['training']['epochs']

    device = set_seed_and_device()

    # Get dataset
    data_filters = get_data_spliters()
    dataset = get_dataset(data_filters=data_filters, train=True)

    train_loader, val_loader = init_loader(dataset)

    model = init_compnet(input_size=dataset.img_size,
                          n_actions=dataset.n_factors).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

    ProgressBar().attach(trainer, output_transform=partial(loss_string, 'loss'))

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def validate(engine):
    #     validator.run(val_loader, epoch_length=2000)

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

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_loss_parameters(engine):
        loss.update_parameters(engine.state.iteration - 1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(val_loader, epoch_length=np.ceil(len(val_loader) * 0.2))
        validator.logger.info(metrics_string(validator.state.metrics))

    # import traceback
    # @trainer.on(Events.EXCEPTION_RAISED)
    # def print_trace(engine):
    #     traceback.print_tb(sys.exc_info()[2])
    #     exit()

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    best_checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent_best_nll',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                                {'model': model})

    # Save every 10 epochs
    periodic_checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent_interval',
        n_saved=epochs//10,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                              periodic_checkpoint, {'model': model})

    # Run the training
    trainer.run(train_loader, max_epochs=epochs)
    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    # Run on test data
    # test_set = load_dataset(batch_size=batch_size, train=False)

    # tester = create_supervised_evaluator(model, metrics, device=device)
    # test_metrics = tester.run(test_set).metrics

    # # Save best model performance and state
    # for metric, value in test_metrics.items():
    #     ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')

    # Save all the periodic
    for name, path in periodic_checkpoint.all_paths:
        ex.add_artifact(path, name)
