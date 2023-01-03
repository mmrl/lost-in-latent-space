import sys
import os
import argparse
import functools

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
parser.add_argument('--plot_recons', action='store_true')
parser.add_argument('--save', type=str, required=True,
                    help='Folder to save the plot')
parser.add_argument('--name', type=str, required=True, help='Plot name')


read_img = lambda folder, file: img_open(os.path.join(folder, file))
read_csv = lambda folder, file: pd.read_csv(os.path.join(folder, file),
                                            index_col=0)


def strip_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_yticks([])


def main(folders, plot_recons, save, name):
    model_ids = [os.path.split(p)[1] for p in folders]

    # Load latent reps
    latent_reps = [read_img(f, 'latent_rep.png') for f in folders]

    fig = plt.figure(constrained_layout=True, figsize=(15, 5))
    gs = plt.GridSpec(2 * (plot_recons + 1), len(model_ids) * 6, figure=fig)

    for i, lr in enumerate(latent_reps):
        ax = fig.add_subplot(gs[2 * (plot_recons): 2 * (plot_recons + 1),
                             i * 6: (i + 1) * 6])
        ax.imshow(lr)
        strip_plot(ax)

    if plot_recons:
        test_recons = [read_img(f, 'test_recons.png') for f in folders]
        train_recons = [read_img(f, 'train_recons.png') for f in folders]

        for i, r in enumerate(train_recons):
            ax = fig.add_subplot(gs[:2, i * 6: i * 6 + 3])
            ax.imshow(r)
            strip_plot(ax)

        for i, r in enumerate(test_recons):
            ax = fig.add_subplot(gs[:2, i * 6 + 3: (i + 1) * 6])
            ax.imshow(r)
            strip_plot(ax)

    os.makedirs(save, exist_ok=True)
    fig.savefig(os.path.join(save, name), bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    args = parser.parse_args()

    folders = args.folders
    save = args.save
    name = args.name
    plot_recons = args.plot_recons

    main(folders, plot_recons, save, name)
