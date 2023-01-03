"""
Models used in the experiments, as a Sacred ingredient.

The models just compose different layers from PyTorch or the ``src'' folder
to create the corresponding architecture. The functions use the parsing
functions in ``utils.py'' to transform a list of layer descriptions into
PyTorch modules. See ``utils.py'' for more details.
"""


import sys
import os
import json
from io import BytesIO

import torch
import torch.nn as nn
from sacred import Ingredient

from .parsing import parse_specs, transpose_specs

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.lgm import LGM
from models.cvae import CascadeVAE
from models.lievae import LieGroupVAE
from models.compnet import CompositionNet, LinearComp, MLPComp, \
                           InterpComp, FixedInterpComp

model = Ingredient('model')


#################################### LGMs #####################################

init_cascade_vae = model.capture(CascadeVAE)
init_lievae = model.capture(LieGroupVAE)

@model.capture
def init_lgm(gm_type, input_size, encoder_layers, latent_size,
             n_cat=0, decoder_layers=None):

    if decoder_layers is None:
        decoder_layers = encoder_layers.copy()

        if gm_type == 'lgm':
            decoder_layers.append(('linear', [latent_size]))
            encoder_layers += [('linear', [2 * latent_size])]
            decoder_input = latent_size

        elif gm_type == 'cascade':
            decoder_layers.append(('linear', [latent_size + n_cat]))
            encoder_layers += [('linear', [2 * latent_size])]
            decoder_input = latent_size + n_cat

        elif gm_type == 'lie':
            total_group_features = sum(latent_size[0])

            decoder_layers.append(('linear', [total_group_features]))
            encoder_layers += [('linear', [total_group_features])]
            decoder_input = total_group_features
        else:
            raise ValueError()

        decoder_layers = transpose_specs(decoder_layers, input_size)

    encoder_layers = parse_specs(input_size, encoder_layers)
    decoder_layers = parse_specs(decoder_input, decoder_layers)

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    if gm_type == 'cascade':
        return init_cascade_vae(latent_size, n_cat, encoder, decoder)

    if gm_type == "lie":
        return init_lievae(*latent_size, encoder, decoder)

    return LGM(latent_size, encoder, decoder)


def load_lgm(path, input_size, device):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']
        architecture['input_size'] = input_size

        # remove cascade vae stuff
        architecture.pop('lmbda', None)
        architecture.pop('disc_warmup', None)

        # remove finetuned stuff
        if 'base_model' in architecture:
            architecture.pop('base_model')
            architecture.pop('composition_op')
            architecture.pop('retrain_decoder')

    lgm = init_lgm(**architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    lgm.load_state_dict(state_dict)

    return lgm.to(device=device).eval()


################################## Supervised ##################################

@model.capture
def init_pred(input_size, layers, n_targets):
    layers += [('linear', [n_targets])]
    layers = parse_specs(input_size, layers)

    return nn.Sequential(*layers)


def load_predictor(path, img_size, n_factors, device):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']

    predictor = init_pred(input_size=img_size,
                          n_targets=n_factors, **architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    predictor.load_state_dict(state_dict)

    return predictor.to(device=device).eval()


################################## CG models ###################################

comp_ops = {'linear': LinearComp,
            'interp': InterpComp,
            'fixint': FixedInterpComp,
            'mlp'   : MLPComp}

@model.capture
def init_compnet(n_actions, input_size, latent_size, composition_op='linear'):
    lgm = init_lgm(input_size=input_size)
    comp_op = comp_ops[composition_op](n_actions, latent_size)

    return CompositionNet(lgm, comp_op)


def load_lgm_from_compnet(path, input_size, device):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']
        del architecture['composition_op']
        architecture['input_size'] = input_size

    lgm = init_lgm(**architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))
        state_dict = {k[4:]: state_dict[k] for k in state_dict
                                           if k.startswith('vae')}
    lgm.load_state_dict(state_dict)

    return lgm.to(device=device).eval()


################################ GT Decoders ##################################

class GaussianNoise(nn.Module):
    def __init__(self, noise):
        super().__init__()
        self.noise = noise

    def forward(self, inputs, random_eval=False):
        if self.training or random_eval:
            eps = torch.randn_like(inputs)
            return inputs + self.noise * eps
        return inputs


@model.capture
def init_decoder(latent_size, decoder_layers, img_size, noise=0.0):
    output_layer = decoder_layers[-1]
    output_layer_name, output_layer_args = output_layer[:2]

    n_channels = img_size[0]

    if output_layer_name == 'tconv':
        output_layer_args = [n_channels] + output_layer_args[1:]
    elif output_layer_name == 'linear':
        output_layer_args = ([n_channels * output_layer_args[0]] +
                             output_layer_args[1:])

    output_layer = [output_layer_name, output_layer_args] + output_layer[2:]

    decoder_layers = decoder_layers[:-1] + [output_layer]
    decoder_layers = parse_specs(latent_size, decoder_layers)

    if noise > 0:
        noise_layer = GaussianNoise(noise)
        decoder_layers.insert(noise_layer, 0)

    return nn.Sequential(*decoder_layers)


def load_decoder(path, dataset, device):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']

    decoder = init_decoder(**architecture, img_size=dataset.img_size,
                           latent_size=dataset.n_factors)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    decoder.load_state_dict(state_dict)

    return decoder.to(device=device).eval()
