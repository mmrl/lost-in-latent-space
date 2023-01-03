"""
Adapted from: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch
"""

import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List
from .initialization import weights_init


class LieGroupVAE(nn.Module):
    def __init__(self,
        subgroup_sizes,
        subspace_sizes,
        encoder,
        decoder,
        lie_alg_init_scale=0.001,
        prob_forward=0.2,
        use_exp=True) -> None:

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lie_layer = LieGroupLayer(subgroup_sizes, subspace_sizes,
                                       lie_alg_init_scale, use_exp)
        self.prob_forward = prob_forward

    @property
    def latent_size(self):
        return sum(self.lie_layer.subgroup_sizes)

    @property
    def subgroup_sizes(self):
        return self.lie_layer.subgroup_sizes

    @property
    def subspace_sizes(self):
        return self.lie_layer.subspace_sizes

    def reset_parameter(self):
        self.apply(weights_init)

    def embed(self, inputs):
        """Embed a batch of data points, x, into their z representations."""
        h = self.encoder(inputs)
        return self.lie_layer(h)[0][2]

    def forward(self, inputs):
        features = self.encoder(inputs)
        latent, g_params = self.lie_layer(features)

        if self.training and np.random.rand() < self.prob_forward:
            output = latent[0]
        else:
            output = latent[2]

        recons = self.decoder(output)
        return recons, latent, g_params


class LieGroupLayer(nn.Module):
    def __init__( self,
            subgroup_sizes: List[int],
            subspace_sizes: List[int],
            lie_alg_init_scale: float= 0.001,
            use_exp: bool=True) -> None:
        super().__init__()

        assert len(subspace_sizes) == len(subgroup_sizes)

        self.subgroup_sizes = subgroup_sizes
        self.subspace_sizes = subspace_sizes
        self.lie_alg_init_scale = lie_alg_init_scale

        # TODO: handle use_exp=False

        self.group_means = nn.ModuleList([])
        self.group_logvars = nn.ModuleList([])
        self.lie_basis = nn.ParameterList([])

        for i, group_size in enumerate(subgroup_sizes):
            # init Gaussian latents
            self.group_means.append(
                    nn.Sequential(
                        nn.Linear(group_size, 4 * group_size),
                        nn.ReLU(),
                        nn.Linear(4 * group_size, subspace_sizes[i])))

            self.group_logvars.append(
                    nn.Sequential(
                        nn.Linear(group_size, 4 * group_size),
                        nn.ReLU(),
                        nn.Linear(4 * group_size, subspace_sizes[i])))

            # init Lie group basis
            mat_dim = int(math.sqrt(group_size))
            assert mat_dim * mat_dim == group_size

            for _ in range(subspace_sizes[i]):
                lie_alg_tmp = init_alg_basis(mat_dim, lie_alg_init_scale)
                self.lie_basis.append(lie_alg_tmp)

    @property
    def group_feats_size(self):
        return sum(self.subgroup_sizes)

    def forward(self, inputs):
        mu, logvar = self.sample_group_params(inputs)
        latents = self.reparam(mu, logvar)
        lie_group_tensor = self.exp_mapping(latents)

        return (inputs, latents, lie_group_tensor), (mu, logvar, self.lie_basis)

    def sample_group_params(self, inputs):
        b_idx = 0
        means_ls, logvars_ls = [], []
        for i, group_size in enumerate(self.subgroup_sizes):
            e_idx = b_idx + group_size
            means_ls.append(self.group_means[i](inputs[:, b_idx:e_idx]))
            logvars_ls.append(self.group_logvars[i](inputs[:, b_idx:e_idx]))
            b_idx = e_idx
        mean, logvar = torch.cat(means_ls, dim=-1), torch.cat(logvars_ls, dim=-1)
        return mean, logvar

    def reparam(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(std)
            return mu.addcmul(std, eps)
        return mu

    def exp_mapping(self, latents):
        lie_group_tensor, b_idx = [], 0

        for i, group_size in enumerate(self.subgroup_sizes):
            mat_dim = int(math.sqrt(group_size))
            e_idx = b_idx + self.subspace_sizes[i]
            if self.subspace_sizes[i] > 1:
                if not self.training:
                    lie_subgroup = val_exp(latents[:, b_idx:e_idx],
                                           self.lie_basis[b_idx:e_idx])
                else:
                    lie_subgroup = train_exp(latents[:, b_idx:e_idx],
                                             self.lie_basis[b_idx:e_idx], mat_dim)
            else:
                lie_subgroup = val_exp(latents[:, b_idx:e_idx],
                                       self.lie_basis[b_idx:e_idx])

            lie_subgroup_tensor = lie_subgroup.view(-1, mat_dim * mat_dim)
            lie_group_tensor.append(lie_subgroup_tensor)
            b_idx = e_idx

        # [b, group_feat_size]
        lie_group_tensor = torch.cat(lie_group_tensor, dim=1)

        return lie_group_tensor


def init_alg_basis(mat_dim, lie_alg_init_scale):
    init_matrix = torch.normal(mean=torch.zeros(1, mat_dim, mat_dim),
                               std=lie_alg_init_scale)
    lie_alg_tmp = Parameter(init_matrix, requires_grad=True)
    return lie_alg_tmp


def train_exp(x, lie_alg_basis_ls, mat_dim):
    # For torch.cat, convert param to tensor.
    lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls ]

    # [1, lat_dim, mat_dim, mat_dim]
    lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[np.newaxis, ...]

    # [1, mat_dim, mat_dim]
    lie_group = torch.eye(mat_dim, dtype=x.dtype).to( x.device)[np.newaxis, ...]

    lie_alg = 0.
    latents_in_cut_ls = [x]
    for masked_latent in latents_in_cut_ls:
        lie_alg_sum_tmp = torch.sum(
            masked_latent[..., None, None] * lie_alg_basis, dim=1)

        lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
        lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
        lie_group = torch.matmul(lie_group, lie_group_tmp)  # [b, mat_dim, mat_dim]

    return lie_group


def val_exp(x, lie_alg_basis_ls):
    # For torch.cat, convert param to tensor.
    lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls ]

    # [1, lat_dim, mat_dim, mat_dim]
    lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[None, ...]
    # [b, lat_dim, mat_dim, mat_dim]
    lie_alg_mul = x[ ..., None, None] * lie_alg_basis

    lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
    lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
    return lie_group
