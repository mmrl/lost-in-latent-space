import sys
import os
import argparse
import functools
import json

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL.Image import open as img_open

parser = argparse.ArgumentParser(description='Summarize results for condition')


parser.add_argument('--model_folders', type=str, nargs='+', dest='folders',
                    help='Folders with saved model results')
parser.add_argument('--recons', action='store_true')
parser.add_argument('--scores', action='store_true')
parser.add_argument('--disent_viz', action='store_true')
parser.add_argument('--latent_reps', action='store_true')
parser.add_argument('--all', action='store_true')
parser.add_argument('--name', type=str, required=True, help='Plot name')
parser.add_argument('--save', type=str, required=True,
                    help='Folder to save the plot')


read_img = lambda folder, file: img_open(os.path.join(folder, file))
read_csv = lambda folder, file: pd.read_csv(os.path.join(folder, file),
                                            index_col=0)


def get_model_name(folder):
    config = os.path.join(folder, 'config.json')

    with open(config) as f:
        config = json.load(f)

    name = config['training']['loss']['name']
    mixing_layer = config['model'].get('mixing_layer', '')

    if name == 'beta-vae':
        name = 'VAE'
    elif name == 'wae-mmd':
        name = 'WAE'
    elif name == 'vectbeta':
        name = "CascadeVAE"
    elif name == 'lie-vae':
        name = "LieGroupVAE"

    if mixing_layer == 'interp':
        name += ' + learned interp'
    elif mixing_layer == 'actalign':
        name += ' + fixed interp'

    return name


def format_scores(scores, disent, model_ids):
    pivoted = []

    for i, s in enumerate(scores):
        s = s.rename(columns={'Metric': 'Model'})
        s['Model'] = model_ids[i]
        s = s.pivot(index='Model', columns='Data', values='model')
        # s = s.reset_index()
        s.columns.name=None
        pivoted.append(s)

    disent = [d.loc['overall_disent']['disentanglement'] for d in disent]
    disent = pd.Series(disent, index=model_ids)

    scores = pd.concat(pivoted)[['Train', 'Test']]
    scores['Disentanglement'] = disent

    return scores


def strip_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_yticks([])


def group_recons(train_recons, test_recons, model_ids):
    n_models = len(train_recons)
    # fig_size = 15, n_models * 3

    # fig, axes= plt.subplots(n_models, 2, figsize=fig_size, squeeze=False)

    # for name, r, ax in zip(model_ids, train_recons, axes[:, 0]):
    #     ax.imshow(r)
    #     strip_plot(ax)
    #     ax.set_title(name + ' (train)', fontdict={'fontsize': 20})

    # for name, r, ax in zip(model_ids, test_recons, axes[:,1]):
    #     ax.imshow(r)
    #     strip_plot(ax)
    #     ax.set_title(name + ' (test)', fontdict={'fontsize': 20})

    fig_size = 15 * n_models, 15

    fig, axes= plt.subplots(2, n_models, figsize=fig_size, squeeze=False)

    for name, r, ax in zip(model_ids, train_recons, axes[0]):
        ax.imshow(r)
        strip_plot(ax)
        ax.set_title(name + ' (train)', fontdict={'fontsize': 20})

    for name, r, ax in zip(model_ids, test_recons, axes[1]):
        ax.imshow(r)
        strip_plot(ax)
        ax.set_title(name + ' (test)', fontdict={'fontsize': 20})

    fig.tight_layout()

    return fig


def group_hinton(matrices, model_ids):
    n_models = len(matrices)
    n_rows = n_models // 5 + 1
    fig_size = 15, 6 * n_rows

    fig = plt.figure(figsize=fig_size)

    for i, (name, m) in enumerate(zip(model_ids, matrices)):
        ax = fig.add_subplot(n_rows, 5, i+1)

        ax.imshow(m)
        strip_plot(ax)
        ax.set_title(name, fontdict={'fontsize':15})

    fig.tight_layout()

    return fig

def group_latent_reps(reps, model_ids):
    n_models = len(reps)
    n_rows = n_models // 4 + 1
    fig_size = 15, 3 * n_rows
    # fig_size = 3 * n_rows, 10

    fig = plt.figure(figsize=fig_size)

    for i, (name, m) in enumerate(zip(model_ids, reps)):
        ax = fig.add_subplot(n_rows, 5, i+1)
        # ax = fig.add_subplot(5, n_rows, i+1)

        ax.imshow(m)
        strip_plot(ax)
        ax.set_title(name)

    fig.tight_layout()

    return fig


def main(folders, name, save, all=False, scores=False, recons=False,
         disent_viz=False, latent_reps=False):

    os.makedirs(save, exist_ok=True)

    model_ids = [get_model_name(p.replace('results', 'sims')) for p in folders]

    # Output scores in latex table
    if all or scores:
        score_dfs = [read_csv(f, 'scores.csv').reset_index() for f in folders]
        dis_dfs = [read_csv(f, 'disentanglement.csv') for f in folders]


        scores = format_scores(score_dfs, dis_dfs, model_ids)

        scores.style.to_latex(os.path.join(save, name + '-scores'))

    if all or recons:
        train_recons = [read_img(f, 'train_recons.png') for f in folders]
        test_recons = [read_img(f, 'test_recons.png') for f in folders]

        fig = group_recons(train_recons, test_recons, model_ids)

        fig.savefig(os.path.join(save, name + '-recons.pdf'),
                          bbox_inches='tight', dpi=300)

    if all or disent_viz:
        hinton_mats = [read_img(f, 'hinton_matrix.png') for f in folders]

        fig = group_hinton(hinton_mats, model_ids)

        fig.savefig(os.path.join(save, name + '-hinton.pdf'),
                    bbox_inches='tight', dpi=300)

    if all or latent_reps:
        reps = [read_img(f, 'latent_rep.png') for f in folders]

        fig = group_latent_reps(reps, model_ids)

        fig.savefig(os.path.join(save, name + '-latent_reps.pdf'),
                    bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    args = parser.parse_args()

    folders = args.folders
    name = args.name
    save = args.save
    all = args.all
    scores = args.scores
    recons = args.recons
    disent_viz = args.disent_viz
    latent_reps = args.latent_reps

    main(folders, name, save, all, scores, recons, disent_viz, latent_reps)
