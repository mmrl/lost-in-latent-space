import torch
import numpy as np
from itertools import product, combinations

def gauss2standard_kl(mean, logvar):
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())


def permute_dims(latent_sample):
    pi = torch.randn_like(latent_sample).argsort(dim=0)
    perm = latent_sample[pi, range(latent_sample.size(1))]
    return perm


def inv_multiquad_sum(x, y, scales, base_scale):
    scales = scales.to(x.device)

    quadratic_term = torch.sum((x - y).pow(2))
    scale_term = scales * base_scale

    return scale_term.div(scale_term + quadratic_term)

    # def kernel(scale):
    #     sxc = scale * base_scale
    #     return sxc / (sxc + quadratic_term)

    # return torch.as_tensor([kernel(s) for s in scales]).sum()


def min_mean_discrepancy(dist1, dist2, kernel, idxs=None):
    intra_dist_idx, cross_dist_idx = idxs if idxs else mmd_idxs(dist1)

    i_idx, j_idx = cross_dist_idx

    dist1_i, dist2_j = dist1[i_idx], dist2[j_idx]

    cross_dist_score = kernel(dist1_i, dist2_j).sum() / len(cross_dist_idx)

    i_idx, j_idx = intra_dist_idx
    dist1_i, dist1_j = dist1[i_idx], dist1[j_idx]
    dist2_i, dist2_j = dist2[i_idx], dist2[j_idx]

    intra_dist_score = (kernel(dist2_i, dist2_j) +
                        kernel(dist1_i, dist1_j)).sum() / len(intra_dist_idx)

    return intra_dist_score - 2 * cross_dist_score


def mmd_idxs(n_samples):
    def zipplist(idxs):
        return tuple(map(list, zip(*idxs)))  # Indices must be in list form

    intra_dist_idx = zipplist(combinations(range(n_samples), r=2))
    cross_dist_idx = zipplist(product(range(n_samples), repeat=2))

    return intra_dist_idx, cross_dist_idx


def calc_hessian_loss(lie_alg_basis_mul_ij):
    hessian_loss = torch.mean(
        torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3]))
    return hessian_loss


def calc_commute_loss(lie_alg_basis_mul_ij):
    lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(
        0, 1, 3, 2)
    commute_loss = torch.mean(
        torch.sum(torch.square(lie_alg_commutator), dim=[2, 3]))
    return commute_loss

def calc_basis_mul_ij(lie_alg_basis_ls_param):
    lie_alg_basis_ls = [alg_tmp * 1. for alg_tmp in lie_alg_basis_ls_param]
    lie_alg_basis = torch.cat(lie_alg_basis_ls,
                              dim=0)[np.newaxis,
                                     ...]  # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
    lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
    lie_alg_basis_outer_mul = torch.matmul(
        lie_alg_basis,
        lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
    hessian_mask = 1. - torch.eye(
        lat_dim, dtype=lie_alg_basis_outer_mul.dtype
    )[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
    lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
    return lie_alg_basis_mul_ij
