import sys
import os
import yaml
from os import path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sacred import Experiment

from ingredients.dataset import dataset, get_lazyloader
from ingredients.training import training, init_metrics, init_loss, \
                                 mse_recons, bern_recons
from ingredients.analysis import analysis, infer, model_score, get_recons, \
                                 get_recons_plot, latent_rep_plot, \
                                 disentanglement_metric, get_factor_idxs, \
                                 create_r_squares, expected_test_dist, \
                                 plot_expected_test_dist, process_discrete, \
                                 project_subspaces

from ingredients.models import load_lgm_from_compnet, load_lgm, load_predictor


an = Experiment(name='analysis', ingredients=[analysis, dataset, training])


an.add_config(score_model=True, plot_recons=True, compute_disent=True,
              plot_latent_rep=True, plot_expected_reps=False,
              root_folder='../data/results', no_cuda=False)

# Configs to run only one analysis
an.add_named_config('score', plot_recons=False, compute_disent=False,
                    plot_latent_rep=False)
an.add_named_config('recons', score_model=False, compute_disent=False,
                    plot_latent_rep=False)
an.add_named_config('disent', score_model=False, plot_recons=False,
                    plot_latent_rep=False)
an.add_named_config('repr', score_model=False, plot_recons=False,
                    compute_disent=False)

# Run all but X
an.add_named_config('noscore', score_model=False)
an.add_named_config('norecons', plot_recons=False)
an.add_named_config('nodisent', compute_disent=False)
an.add_named_config('norepr', plot_latent_rep=False)

# Run either performance or latent representation plots
an.add_named_config('perfm', compute_disent=False, plot_latent_rep=False)
an.add_named_config('latent', score_model=False, plot_recons=False)

training.add_named_config('recons_mse', metrics=[mse_recons])
training.add_named_config('recons_bern', metrics=[bern_recons])


def is_generative(setting):
    return setting in ['composition', 'unsupervised', 'recons']


@an.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def load_model(setting, dataset, model_folder, device):
    if setting == 'unsupervised':
        return load_lgm(model_folder, dataset.img_size, device)
    elif setting == 'supervised':
        return load_predictor(model_folder, dataset.img_size,
                              dataset.n_factors, device)
    return load_lgm_from_compnet(model_folder, dataset.img_size, device)


@an.automain
def main(_config, model_id, exp_folder, score_model, plot_recons,
         compute_disent, plot_latent_rep, plot_expected_reps, root_folder):

    print('Running analysis for model {}.'.format(model_id))

    device = set_seed_and_device()

    model_folder = path.join(exp_folder, str(model_id))

    with open(path.join(model_folder, 'config.json')) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Extract dataset conifg
    dataset_config = configs['dataset']

    dataset = dataset_config['dataset']
    setting = dataset_config['setting']
    condition = dataset_config.get('condition', None)
    variant = dataset_config.get('variant', None)
    modifiers = dataset_config.get('modifiers', None)

    # Extract model variant
    model_cfg = configs['model']
    model_type = model_cfg['gm_type']

    if is_generative(setting):
        loss = configs['training']['loss']['params']['reconstruction_loss']
    else:
        loss = configs['training']['loss']['name']

    # Create results folder
    # results_folder = path.join(root_folder, setting, str(model_id))
    results_folder = path.normpath(exp_folder).split(path.sep)
    results_folder[results_folder.index('sims')] = 'results'
    results_folder = path.join(*results_folder, str(model_id))

    os.makedirs(results_folder, exist_ok=True)

    print('Loading model and dataset...')

    # model = load_composer(model_folder, model_id, device)
    dataset = get_lazyloader(dataset, condition, variant, modifiers)
    model = load_model(setting, dataset, model_folder, device)

    print('Done.')

    if score_model:
        print('Computing model scores...')
        if loss == 'mse':
            if is_generative(setting):
                metrics = init_metrics([mse_recons])
            else:
                metrics = create_r_squares(dataset.factors)
        elif loss == 'bce':
            metrics = init_metrics([bern_recons])
        else:
            raise ValueError("Unsuported loss {}.".format(loss))

        print('Scoring training data...')
        train_score = model_score(model,
                dataset.get_unsupervised(train=True), metrics=metrics)

        print('Done.')

        test_data = dataset.get_unsupervised(train=False)

        if test_data is not None:
            print('Test data provided. Scoring model on OOD samples...')

            test_score = model_score(model,
                    dataset.get_unsupervised(train=False), metrics=metrics)

            all_scores = pd.concat([train_score, test_score],
                                   keys=['Train', 'Test'], names=['Data'])

            print('Done.')
        else:
            print('No test data provided')
            all_scores = train_score.reset_index()

        all_scores.to_csv(path.join(results_folder, 'scores.csv'))

        print('Saving results.\nDone.')

    train_data = dataset.get_supervised(train=True, pred_type='reg')
    test_data = dataset.get_supervised(train=False, pred_type='reg')

    if is_generative(setting) and plot_recons:
        print('Generating reconstruction examples for training data...')

        train_recons = get_recons(model, train_data, loss=loss)[:2]
        train_recons_fig = get_recons_plot(train_recons)
        train_recons_fig.savefig(path.join(results_folder,
                                 'train_recons.png'), bbox_inches='tight')

        print('Done.')

        if test_data is not None:
            print('Generating reconstruction examples for test data...')

            test_recons = get_recons(model, test_data, loss=loss)[:2]
            test_recons_fig = get_recons_plot(test_recons)
            test_recons_fig.savefig(path.join(results_folder,
                                    'test_recons.png'), bbox_inches='tight')

            print('Done.')

    if compute_disent or plot_latent_rep:
        print("Computing model outputs...")

        train_proj = infer(model, train_data)

        if test_data is None:
            test_proj = None, None
        else:
            test_proj = infer(model, test_data)

        if hasattr(model, 'n_cat'):
            train_proj, test_proj = process_discrete(train_proj, test_proj, model)

        if model_type == 'lie':
            train_proj, test_proj, pca_results = project_subspaces(train_proj,
                                                                   test_proj,
                                                                   model)

        print("Done.")

        if is_generative(setting):
            print("Computing disentanglement...")
            daf_results = disentanglement_metric(model, train_data, test_data,
                projections=(train_proj, test_proj))
            dim1, dim2 = get_factor_idxs(daf_results)
        else:
            factor1 = _config['analysis']['factor1']
            factor2 = _config['analysis']['factor2']
            dim1 = dataset.factors.index(factor1)
            dim2 = dataset.factors.index(factor2)

        print("Done.")

    if is_generative(setting) and compute_disent:
        print('Saving disentanglement scores...')
        results = daf_results.score2df()
        disent_scores, complete_scores, overall_disent = results

        disent_scores.loc['overall_disent'] = overall_disent, None

        disent_scores.to_csv(path.join(results_folder,
                                       'disentanglement.csv'))
        complete_scores.to_csv(path.join(results_folder,
                                         'completeness.csv'))
        h, w = daf_results.hinton_matrix.T.shape
        hinton_matrix, ax = plt.subplots(figsize=(2 * h,3 * w))

        daf_results.plot_hinton(ax=ax)
        hinton_matrix.savefig(path.join(results_folder, 'hinton_matrix.png'),
                              dpi=300, bbox_inches='tight')
        print('Done.')

    if plot_latent_rep or plot_expected_reps:
        print('Plotting latent representation...')

        latent_rep = latent_rep_plot(model, train_data, dim1, dim2,
                                     test_data=test_data,
                                     train_proj=train_proj,
                                     test_proj=test_proj)

        latent_rep.savefig(path.join(results_folder, 'latent_rep.png'),
                           bbox_inches='tight', dpi=300)
        print('Done.')

    if plot_expected_reps:
        print('Plotting expected distributions for left out combinations')

        factor1 = _config['analysis']['factor1']
        factor2 = _config['analysis']['factor2']

        f1 = dataset.factors.index(factor1)
        f2 = dataset.factors.index(factor2)

        exp_mean, exp_std = expected_test_dist(train_proj, test_proj,
                                               dim1, dim2, f1, f2)

        plot_expected_test_dist(exp_mean, exp_std, 100, latent_rep)

        latent_rep.savefig(path.join(results_folder, 'latent_rep+expected.png'),
                           bbox_inches='tight', dpi=300)

    print('Analysis finished')
