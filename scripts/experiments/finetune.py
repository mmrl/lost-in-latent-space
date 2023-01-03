"""
Experiment script to train ground truth decoders the disentanglement datasets.

As with the standard end-to-end models, the decoders are trained to reconstruct
unseen images of novel combinations of generative factors. Contratry to the standard
models, the ground truth disentanglement is fed into the decoders as opposed to being
learned by them throught trianing.

The architectures defined are the same decoders used by the standard models, plus
some additions containing modifactions to depth and adding layers such as
        batch normalization.

To run:
    cd <experiment-root>/scripts/
    python -m experiment.decoders with dataset.<option> model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred documentation
[https://sacred.readthedocs.io/en/stable/]
"""

import sys
import os
import numpy as np
import torch
import json
from io import BytesIO

from sacred import Experiment
from sacred.observers import FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, \
                                get_data_spliters
from ingredients.models import model, init_lgm, load_lgm_from_compnet
from ingredients.training import training, init_loss, init_metrics, \
                                 init_optimizer, ModelCheckpoint, \
                                 init_loader

import configs.training as train_params
import configs.vaes as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer


# Set up experiment
ex = Experiment(name='finetune', ingredients=[dataset, model, training])

# Observers
ex.observers.append(FileStorageObserver.create('../data/sims/finetune'))

# General configs
ex.add_package_dependency('torch', torch.__version__)
ex.add_config(no_cuda=False, save_folder='../data/temp/finetune')


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
dataset.add_config(dataset='dsprites', setting='unsupervised', shuffle=True)

# dataset.add_named_config('dsprites', dataset='dsprites')
# dataset.add_named_config('shapes3d', dataset='shapes3d', color_mode='rgb')
# dataset.add_named_config('mpi3d', dataset='mpi3d', version='real')


# Training configs
# training.add_config(loss=reconstruction_loss, metrics=[reconstruction_loss])
training.config(train_params.vae)

# Models
model.add_config(base_model=None, retrain_decoder=False)
model.named_config(model_params.higgins)
model.named_config(model_params.burgess)
model.named_config(model_params.kim)
model.named_config(model_params.abdi)


@model.capture
def init_model(img_size, base_model, retrain_decoder,device):
    model = load_lgm_from_compnet(base_model,device)

    if retrain_decoder:
        # model.decoder.reset_parameters()
        for p in model.encoder.parameters():
            p.requires_grad = False

        for m in model.decoder.children():
            try:
                m.reset_parameters()
            except AttributeError:
                pass

    else:
        for p in model.decoder.parameters():
            p.requires_grad = False

        for m in model.encoder.children():
            try:
                m.reset_parameters()
            except AttributeError:
                pass

    return model


# Experiment run
@ex.automain
def main(_config):
    epochs = _config['training']['epochs']

    device = set_seed_and_device()

    # Get dataset
    data_filters = get_data_spliters()
    dataset = get_dataset(data_filters=data_filters, train=True)

    train_loader, val_loader = init_loader(dataset)

    model = init_model(img_size=dataset.img_size,device=device).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

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
