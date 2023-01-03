"""
Experiment script to train standard autoencoder models on disentanglement datasets.

The models are trained on the standard reconstruction task. To test generalisation,
images corresponding to some generative factor combinations are excluded from the
training set. These are removed systematically to test different generalisation
settings.

The architectures and training objectives used are similar to the ones studied in
Locatello et al., 2019 (with the exception of DIP-VAE):
    1. VAE (Kingma & Welling,2014; Rezende & Mohammed 2014)
    2. $\beta$-VAE w/o CCI (Higgins et al, 2017; Burgess et al, 2018)
    3. FactorVAE (Kim & Mnih, 2019)

Note that to train a model on a dataset in this way, we need the generative factors
to be defined explicitly, otherwise we cannot exclude particular combinations.

To run:
    cd <experiment-root>/scripts/
    python -m experiment.disent with dataset.<option> model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred documentation
[https://sacred.readthedocs.io/en/stable/]

"""


import sys
import numpy as np
import torch
from functools import partial

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver, FileStorageObserver
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, get_data_spliters
from ingredients.models import model, init_lgm
from ingredients.training import training, ModelCheckpoint, init_metrics, \
                                 init_optimizer, init_lr_scheduler, \
                                 init_loader, init_loss


import configs.training as train_params
import configs.vaes as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer

# Set up experiment
ex = Experiment(name='disent', ingredients=[dataset, model, training])

# Required for ProgressBar to work properly
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Observers
ex.observers.append(FileStorageObserver.create('../data/sims/disent'))
# ex.observers.append(MongoObserver.create(url='127.0.0.1:27017',
#                                          db_name='disent'))

# General configs
ex.add_config(no_cuda=False, save_folder='../data/sims/disent/temp')
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


def loss_string(name, output):
    return {f"{name.upper()}": f"{output:,.2f}"}


def metrics_string(metrics):
    output = ['Validation:']
    for k, v in metrics.items():
        output.append(f"{k.upper()}={v:,.2f}")
    return ' '.join(output)

# Dataset configs
dataset.add_config(setting='unsupervised', shuffle=True)

# Training configs
training.add_config(scheduler=None)
training.config(train_params.vae)
training.named_config(train_params.beta)
training.named_config(train_params.cci)
training.named_config(train_params.factor)
training.named_config(train_params.waegan)
training.named_config(train_params.waemmd)
training.named_config(train_params.bsched)
training.named_config(train_params.banneal)
training.named_config(train_params.cascade)
training.named_config(train_params.step_lr)
training.named_config(train_params.lievae)

# Model configs
model.named_config(model_params.higgins)
model.named_config(model_params.burgess)
model.named_config(model_params.burgess_v2)
model.named_config(model_params.mpcnn)
model.named_config(model_params.mathieu)
model.named_config(model_params.kim)
model.named_config(model_params.kim_bn)
model.named_config(model_params.abdi)
model.named_config(model_params.montero)
model.named_config(model_params.tolstikhin)
model.named_config(model_params.watters)
model.named_config(model_params.sbd3)
model.named_config(model_params.sbd4)
model.named_config(model_params.cascade)
model.named_config(model_params.liegroup)

# Run experiment
@ex.automain
def main(_config):
    epochs = _config['training']['epochs']

    device = set_seed_and_device()
    # Load data
    data_filters = get_data_spliters()
    dataset = get_dataset(data_filters=data_filters, train=True)

    training_loader, validation_loader = init_loader(dataset)

    # Init model
    img_size = training_loader.dataset.img_size
    model = init_lgm(input_size=img_size).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # init schedular
    scheduler = init_lr_scheduler(optimizer)

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

    ProgressBar().attach(trainer, output_transform=partial(loss_string, 'loss'))

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.terminate()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_loss_parameters(engine):
        loss.update_parameters(engine.state.iteration - 1)

    if scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                lambda _: scheduler.step())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        ex.log_scalar('training_loss', tracer.loss[-1])
        tracer.loss.clear()

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_loader,
                      epoch_length=np.ceil(len(validation_loader) * 0.2))
        validator.logger.info(metrics_string(validator.state.metrics))

    # @validator.on(Events.EPOCH_COMPLETED)
    # def log_validation(engine):
    #     for metric, value in engine.state.metrics.items():
    #         ex.log_scalar('val_{}'.format(metric), value)

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
    trainer.run(training_loader, max_epochs=epochs)
    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    # # Save best model performance and state
    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')

    # Save all the periodic
    for name, path in periodic_checkpoint.all_paths:
        ex.add_artifact(path, name)
