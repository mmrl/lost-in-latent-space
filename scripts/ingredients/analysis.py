import sys
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn import functional as F
from sacred import Ingredient
from ignite.contrib.metrics.regression import R2Score
from torch.utils.data.dataloader import DataLoader
from ignite.engine import create_supervised_evaluator
from sklearn.decomposition import PCA

if '../src' not in sys.path:
    sys.path.append('../src')

from analysis.testing import infer
from analysis.metrics import DAF
from analysis.hinton import hinton


torch.set_grad_enabled(False)
sns.set(color_codes=True)
sns.set_style("white", {'axes.grid': False})
plt.rcParams.update({'font.size': 11})

analysis = Ingredient('analysis')


def create_r_squares(output_names):
    n_out = len(output_names)

    def slice_idx(i):
        def transform(output):
            y_pred, y = output
            return y_pred[:, i], y[:, i]
        return transform

    return {'{}'.format(n): R2Score(slice_idx(i))
            for i, n in enumerate(output_names)}


def discrete2con(z, n_cat):
    z_cat = F.softmax(z[:, -n_cat:], dim=-1).argmax(dim=1)
    z_cat = z_cat.to(dtype=torch.float)
    # normalise to [-1, 1]
    z_cat = 2 * z_cat / (z_cat.max() - z_cat.min()) - 1.0
    return torch.cat([z[:, :-n_cat], z_cat[..., None]], dim=-1)


def process_discrete(train_proj, test_proj, model):
    z_train, gt_train = train_proj
    z_train = discrete2con(z_train, model.n_cat)
    train_proj = z_train, gt_train

    if test_proj[0] is not None:
        z_test, gt_test = test_proj
        z_test = discrete2con(z_test, model.n_cat)
        test_proj = z_test, gt_test

    return train_proj, test_proj


@dataclass
class PCAResults:
    coeff: np.ndarray
    explained_var: np.ndarray
    singular_values: np.ndarray


def project_subspaces(train_data, test_data, model):
    subspace_sizes = model.subspace_sizes
    subgroup_sizes = model.subgroup_sizes

    train_latents = train_data[0].cpu().numpy()
    if test_data[0] is not None:
        test_latents = test_data[0].cpu().numpy()
    else:
        test_latents = None

    def _project(i):
        g_size, sp_size = subgroup_sizes[i], subspace_sizes[i]
        n_subspaces = g_size // sp_size

        pca = PCA(n_components=sp_size, copy=False)

        train_proj, test_proj, pca_results = [], [], []
        for s_idx in range(n_subspaces):
            X = train_latents[:, s_idx * sp_size: (s_idx + 1) * sp_size]
            proj = pca.fit_transform(X)
            train_proj.append(proj[:, 0:1])

            pca_results.append(PCAResults(pca.components_,
                                          pca.explained_variance_,
                                          pca.singular_values_))

            if test_latents is not None:
                X = test_latents[:, s_idx * sp_size: (s_idx + 1) * sp_size]
                proj = pca.transform(X)
                test_proj.append(proj[:, 0:1])

        train_proj = np.concatenate(train_proj, axis=1)
        train_proj = train_data[0].new_tensor(train_proj)

        if len(test_proj) == 0:
            test_proj = None
        else:
            test_proj = np.concatenate(test_proj, axis=1)
            test_proj = test_data[0].new_tensor(test_proj)

        return train_proj, test_proj, pca_results

    train_proj, test_proj, pca_results = [], [], []

    for i in range(len(subgroup_sizes)):
        sp_train, sp_test, sp_pca = _project(i)

        train_proj.append(sp_train)
        if sp_test is not None:
            test_proj.append(sp_test)
        pca_results.append(sp_pca)

    train_proj = torch.cat(train_proj, dim=1)
    if len(test_proj) > 0:
        test_proj = torch.cat(test_proj, dim=1)
    else:
        test_proj = None

    return (train_proj, train_data[1]), (test_proj, test_data[1]), pca_results


# def pca_plos(results):


@analysis.capture
def model_score(model, data, metrics, model_name=None, device=None, *kwargs):
    dataloader_args = {'batch_size': 120, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    if model_name is None:
        model_name = 'model'
    if device is None:
        device = next(model.parameters()).device

    engine = create_supervised_evaluator(model, metrics, device)
    metrics = engine.run(loader).metrics

    index = pd.Index(metrics.keys(), name='Metric')
    scores = pd.Series(metrics.values(), index=index, name=model_name)

    return scores


@analysis.capture
def expected_test_dist(training_data, test_data, dim1, dim2, f1, f2):
    training_reps, training_factors = training_data
    test_factors = test_data[1]

    def var_mean(dim, f_idx):
        means, stds = [], []

        for f_val in torch.unique(training_factors[:, f_idx]):
            mask = training_factors[:, f_idx] == f_val
            val_reps = training_reps[mask, dim]

            std, mean = torch.std_mean(val_reps)

            means.append(mean.item())
            stds.append(std.item())

        return torch.tensor(means), torch.tensor(stds)

    dim1_means, dim1_std = var_mean(dim1, f1)
    dim2_means, dim2_std = var_mean(dim2, f2)

    combination_means, combination_std = [], []

    f1_unique = torch.unique(training_factors[:, f1])
    f2_unique = torch.unique(training_factors[:, f2])

    all_combinations  = product(enumerate(f1_unique), enumerate(f2_unique))

    for (f1_val_idx, f1_val), (f2_val_idx, f2_val) in all_combinations:
        mask = (test_factors[:, f1] == f1_val) & (test_factors[:, f2] == f2_val)

        if torch.any(mask):
            combination_means.append(torch.tensor([dim1_means[f1_val_idx],
                                                   dim2_means[f2_val_idx]]))

            combination_std.append(torch.tensor([dim1_std[f1_val_idx],
                                                 dim2_std[f2_val_idx]]))

    return torch.stack(combination_means), torch.stack(combination_std)


def plot_expected_test_dist(exp_mean, exp_std, n_samples, fig):
    ax = fig.get_axes()[0]

    exp_mean = exp_mean.unsqueeze_(1).expand(-1, n_samples, -1)
    exp_std = exp_std.unsqueeze_(1).expand(-1, n_samples, -1)
    eps = torch.randn_like(exp_mean)

    samples = exp_mean.addcmul(eps, exp_std)

    for i in range(samples.size(1)):

        x_joint = samples[:, i, 1].cpu().numpy()
        y_joint = samples[:, i, 0].cpu().numpy()

        color = (0.85, 0.0, 0.0)  # Not so bright red

        ax.scatter(x_joint, y_joint, color=color, alpha=0.1, marker='x')


@analysis.capture
def get_recons(model, data, n_recons=10, loss='bce', **kwargs):
    dataloader_args = {'batch_size': n_recons, 'shuffle': True,
                       'pin_memory': True}
    dataloader_args.update(kwargs)

    inputs, targets = next(iter(DataLoader(data, **dataloader_args)))

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        recons = model(inputs.to(device=device))
        if isinstance(recons, tuple):
            recons = recons[0]

        if loss == 'bce':
            recons = recons.sigmoid()
        else:
            recons = recons.clamp(0, 1)

        recons = recons.cpu()

    return inputs, recons, targets


@analysis.capture
def get_recons_plot(data, no_recon_labels=False, axes=None):
    inputs, recons = data

    batch_size = len(inputs)

    if axes is None:
        fig, axes = plt.subplots(2, batch_size, figsize=(2 * batch_size, 4))
    else:
        fig = None

    images = np.stack([inputs.numpy(), recons.numpy()])

    for j, (examp_imgs, ylab) in enumerate(zip(images, ['original', 'model'])):
        for i, img in enumerate(examp_imgs):
            if np.prod(img.shape) == 3 * 64 * 64:
                axes[j, i].imshow(img.reshape(3, 64, 64).transpose(1, 2, 0))
            else:
                axes[j, i].imshow(img.reshape(1, 64, 64).transpose(1, 2, 0),
                                  cmap='Greys_r')
            # if i == 0:
            #     axes[j, i].set_ylabel(ylab, fontsize=28)

    for ax in axes.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if not no_recon_labels:
        axes[0, 0].set_ylabel('input', fontsize=20)
        axes[1, 0].set_ylabel('recons', fontsize=20)

    return fig


@analysis.capture
def infer(model, data, **kwargs):
    dataloader_args = {'batch_size': 128, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in loader:
            x = x.to(device=device)
            # z = model(x)
            z = model.embed(x)

            if isinstance(z, tuple):
                z = z[1]

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets


@analysis.capture
def get_factor_idxs(daf_results, factor1, factor2):
    # Find the corresponding dimension and assign
    f1_idx = daf_results.gt_names.index(factor1)
    f2_idx = daf_results.gt_names.index(factor2)

    dim1 = daf_results.sort_idx[f1_idx]
    dim2 = daf_results.sort_idx[f2_idx]

    return dim1, dim2


@analysis.capture
def latent_rep_plot(model, train_data, dim1, dim2, factor1, factor2,
                    test_data=None, train_proj=None, test_proj=None,
                    joint_palette='dim1'):
    # get encoded values
    if train_proj is not None:
        z, targets = train_proj
    else:
        z, targets = infer(model, train_data)

    # z += 0.01 * np.random.randn(*z.shape)

    if test_proj[0] is not None:
        if test_proj:
            z_test, test_targets = test_proj
        else:
            z_test, test_targets = infer(model, test_data)

        # z_test += 0.01 * np.random.randn(*z_test.shape)

        train_alpha = 0.1
    else:
        train_alpha = 1.0

    # Set plot
    fig = plt.figure(figsize=(10, 5))

    # dim1_ax = fig.add_subplot(3, 5, (5, 10))  # y axis
    # dim2_ax = fig.add_subplot(3, 5, (11, 14))  # x axis
    # joint_ax = fig.add_subplot(3, 5, (1, 9), sharex=dim2_ax)
    joint_ax = fig.add_subplot(111)

    # Remove unnecessary axes ticks and spines
    # plt.setp(joint_ax.get_xticklabels(), visible=False)
    # plt.setp(dim2_ax.get_yticklabels(), visible=False)

    # dim1_ax.set_xticklabels([])
    # dim1_ax.set_yticklabels([])

    # for ax in [dim1_ax, dim2_ax, joint_ax]:
    for ax in [joint_ax]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set titles
    # dim1_ax.set_title(factor1)
    # dim2_ax.set_xlabel(factor2)
    joint_ax.set_ylabel(factor1.replace("_", " "), fontsize=36)
    joint_ax.set_xlabel(factor2.replace("_", " "), fontsize=36)
    joint_ax.tick_params(axis='both', which='major', labelsize=28)

    # Palettes for the plots
    n_colors = len(train_data.unique_values[factor1])
    dim1_palette = sns.color_palette("tab10", n_colors)

    n_colors = len(train_data.unique_values[factor2])
    dim2_palette = sns.color_palette("tab10", n_colors)

    def plot_marginal(z_proj, gf_codes, unique, palette, ax,
                      flip=False, fill=True):
        for c in unique:
            c_instances = z_proj[gf_codes == c]
            if len(c_instances) > 0:
                if flip:
                    sns.kdeplot(y=c_instances, color=palette[c],
                                ax=ax, fill=fill)
                else:
                    sns.kdeplot(x=c_instances, color=palette[c],
                                ax=ax, fill=fill)

    def plot_joint(z1_proj, z2_proj, gf1_codes, gf2_codes,
                   unique1, unique2, alpha=1.0, is_train=True):
        for c1, c2 in product(unique1, unique2):
            idx = (gf1_codes == c1) & (gf2_codes == c2)
            x_joint = z2_proj[idx]
            y_joint = z1_proj[idx]

            # if joint_palette == 'dim1':
            #     color = dim1_palette[c1]
            # else:
            #     color = dim2_palette[c2]
            if is_train:
                # color = 'black'
                color = dim1_palette[c1]
            else:
                color = (0.85, 0.0, 0.0) # Not so bright red

            if len(x_joint) > 1:
                sns.kdeplot(x=x_joint, y=y_joint, color=color,
                            ax=joint_ax, alpha=alpha)
            elif len(x_joint) == 1:
                joint_ax.scatter(x_joint, y_joint, color=color,
                                alpha=alpha, marker='x')

    # training data projection
    z_dim1, z_dim2 = z[:, dim1], z[:, dim2]

    gt1_idx = train_data.factors.index(factor1)
    gt2_idx = train_data.factors.index(factor2)

    # ground truth codes (i.e. class index for each dimension)
    gf1_codes = train_data.factor_classes[:, gt1_idx]
    gf2_codes = train_data.factor_classes[:, gt2_idx]

    gf1_unique = np.unique(gf1_codes)
    gf2_unique = np.unique(gf2_codes)

    # plot_marginal(z_dim1, gf1_codes, gf1_unique,
    #               dim1_palette, dim1_ax, True, train_alpha == 1)
    # plot_marginal(z_dim2, gf2_codes, gf2_unique,
    #               dim2_palette, dim2_ax, False, train_alpha == 1)

    plot_joint(z_dim1, z_dim2, gf1_codes, gf2_codes,
               gf1_unique, gf2_unique)

    if test_data is not None:
        z_test_dim1, z_test_dim2 = z_test[:, dim1], z_test[:, dim2]

        gf1_codes = test_data.factor_classes[:, gt1_idx]
        gf2_codes = test_data.factor_classes[:, gt2_idx]

        gf1_unique = np.unique(gf1_codes)
        gf2_unique = np.unique(gf2_codes)

        # plot_marginal(z_test_dim1, gf1_codes, gf1_unique,
        #               dim1_palette, dim1_ax, True)
        # plot_marginal(z_test_dim2, gf2_codes, gf2_unique,
        #               dim2_palette, dim2_ax, False)

        plot_joint(z_test_dim1, z_test_dim2, gf1_codes, gf2_codes,
                   gf1_unique, gf2_unique, is_train=False)

    # joint_ax.set_ylim([-0.5, 1.5])
    # joint_ax.set_xlim([-0.1, 1.3])

    # dim1_ax.set_xlabel("")
    # dim2_ax.set_ylabel("")
    # joint_ax.set_xlabel("")
    # joint_ax.set_ylabel("")

    fig.tight_layout()

    return fig


@analysis.capture
def disentanglement_metric(model, train_data, test_data=None, projections=None,
                           method='lasso', assignment='optimal',
                           method_args=None):

    daf = DAF(train_data, test_data=test_data, method=method,
              assignment=assignment, method_kwargs=method_args)

    daf_results = daf(model, projections)

    return daf_results
