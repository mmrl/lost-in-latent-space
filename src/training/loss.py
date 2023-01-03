"""
Loss functions for training the models

The losses used to trained disentangled models are here. Most of the ones
analyzed in Locatello et al, 2020 are included, except DIP-VAE I and II.
Metrics for evaluation the models are also included.

Note: This might not be the best way to implement this. I have tried to keep
architectures, loss functions etc. as separated as possible. However, this
means that controlling the behaviour of some functions during training is
harder to achieve.
"""

import math
import torch
import torch.nn as nn
from functools import partial
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss, cross_entropy
from torch.nn.modules.loss import _Loss

from .optimizer import init_optimizer
from .math import gauss2standard_kl, inv_multiquad_sum, min_mean_discrepancy, \
                  permute_dims, mmd_idxs, calc_hessian_loss, calc_commute_loss, \
                  calc_basis_mul_ij


class AELoss(_Loss):
    """
    Base autoencoder loss
    """
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(reconstruction_loss))
        else:
            recons_loss = reconstruction_loss

        self.recons_loss = recons_loss

    def forward(self, input, target):
        reconstruction, *latent_terms = input
        # target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        recons_loss /= target.size(0)

        latent_term = self.latent_term(*latent_terms)

        return recons_loss + latent_term

    def latent_term(self):
        raise NotImplementedError()


class InfoCascadeLoss(AELoss):
    def __init__(self, reconstruction_loss='bce', n_cont=6,
                 beta=(1.0, 10.0), warmup=20000):
        super().__init__(reconstruction_loss=reconstruction_loss)

        self.beta_l = beta[0]
        self.beta_h = beta[1]
        self.warmup = warmup
        self.beta = torch.full((n_cont,), self.beta_h).squeeze_()

    def latent_term(self, z_sample, z_params):
        (mu, logvar), _ = z_params
        beta = self.beta.to(device=mu.device, dtype=mu.dtype)
        kl_div = beta * gauss2standard_kl(mu, logvar)
        return  kl_div.sum() / z_sample.size(0)

    def update_parameters(self, step):
        if ((step + 1) % self.warmup) == 0:
            i = step // self.warmup
            if i < len(self.beta):
                self.beta[i] = self.beta_l

class GaussianVAELoss(AELoss):
    """
    This class implements the Variational Autoencoder loss with Multivariate
    Gaussian latent variables. With defualt parameters it is the one described
    in "Autoencoding Variational Bayes", Kingma & Welling (2014)
    [https://arxiv.org/abs/1312.6114].

    When $\beta>1$ this is the the loss described in $\beta$-VAE: Learning
    Basic Visual Concepts with a Constrained Variational Framework",
    Higgins et al., (2017) [https://openreview.net/forum?id=Sy2fzU9gl]
    """
    def __init__(self, reconstruction_loss='bce', beta=1.0,
                 beta_schedule=None):
        super().__init__(reconstruction_loss)
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.anneal = 1.0

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = gauss2standard_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.anneal * self.beta * kl_div

    def update_parameters(self, step):
        if self.beta_schedule is not None:
            steps, schedule_type, min_anneal = self.beta_schedule
            delta = 1 / steps

            if schedule_type == 'anneal':
                self.anneal = max(1.0 - step * delta, min_anneal)
            elif schedule_type == 'increase':
                self.anneal = min(min_anneal + delta * step, 1.0)


class CCIVAE(AELoss):
    """
    $\beta$-VAE trained with a constrained capacity increase loss (CCI-VAE).
    As in Burgess et al, 2018[https://arxiv.org/pdf/1804.03599.pdf%20].

    This loss slowly increases the strangth of the prior to force the models
    to "kill" unneccesary units in the latent representation.
    """
    def __init__(self, reconstruction_loss='bce', gamma=100.0,
                 capacity=0.0, capacity_schedule=None):
        super().__init__(reconstruction_loss)

        self.gamma = gamma
        self.capacity = capacity
        self.capacity_schedule = capacity_schedule

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = gauss2standard_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.gamma * (kl_div - self.capacity).abs()

    def update_parameters(self, step):
        if self.capacity_schedule is not None:
            cmin, cmax, increase_steps = self.capacity_schedule
            delta = (cmax - cmin) / increase_steps

            self.capacity = min(cmin + delta * step, cmax)


class FactorLoss(AELoss):
    """
    FactorVAE loss as described in Disentangling by Factorizing,
    Kim & Mnih (2019) [https://arxiv.org/pdf/1802.05983.pdf].

    This loss uses adversarial training to minimize the total correlation
    while avoiding any penalization to the mutual information between input
    and latent codes (unlike $\beta-VAE).
    """
    def __init__(self, reconstruction_loss='bce',
                 gamma=10.0, gamma_schedule=None,
                 disc_args=None, optim_kwargs=None):
        super().__init__(reconstruction_loss)
        self.gamma = gamma
        self.gamma_schedule = gamma_schedule
        self.anneal = 1.0

        default_optim_kwargs = {'optimizer': 'adam', 'lr': 1e-4,
                                'betas': (0.5, 0.9)}
        if optim_kwargs is not None:
            optim_kwargs = {**default_optim_kwargs, **optim_kwargs}
        else:
            optim_kwargs = default_optim_kwargs

        # TODO: make this a parameter of the loss
        # self.disc = nn.Sequential(
        self.disc = list([
            # nn.Linear(8, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2),
        ])

        self.optim = partial(init_optimizer, **optim_kwargs)
        self._batch_samples = None

    @property
    def disc_device(self):
        return next(self.disc.parameters()).device

    def initialize(self, latent):
        latent_size = latent.shape[-1]
        in_features = self.disc[1].in_features
        linear = nn.Linear(latent_size, in_features)

        self.disc = nn.Sequential(linear, *self.disc).to(latent.device)
        self.optim = self.optim(params=self.disc.parameters())

    def train(self, mode=True):
        self.disc.train(mode)
        for p in self.disc.parameters():
            p.requires_grad = mode

    def eval(self):
        self.train(False)

    def latent_term(self, z_sample, z_params):
        if isinstance(self.disc, list):
            self.initialize(z_sample)

        self.eval()
        mu, logvar = z_params

        z_sample1, z_sample2 = z_sample.chunk(2, 0)
        mu1, mu2 = mu.chunk(2, 0)
        logvar1, logvar2 = logvar.chunk(2, 0)

        kl_div = gauss2standard_kl(mu1, logvar1).sum()
        kl_div /= z_sample1.size(0)

        log_z_ratio = self.disc(z_sample1)
        total_correlation = (log_z_ratio[:, 0] - log_z_ratio[:, 1]).mean()

        # print(total_correlation)
        self._batch_samples = z_sample1.detach(), z_sample2.detach()

        return kl_div + self.anneal * self.gamma * total_correlation

    def update_parameters(self, step):
        # update anneal value
        if self.gamma_schedule is not None:
            steps, min_anneal = self.gamma_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)

        if self._batch_samples is None:
            return

        # Train discriminator
        self.train()
        z1_samples, z2_samples = self._batch_samples
        self._batch_samples = None

        z_perm = permute_dims(z2_samples)

        log_ratio_z = self.disc(z1_samples)
        log_ratio_z_perm = self.disc(z_perm)

        ones = torch.ones(z_perm.size(0), dtype=torch.long,
                          device=z_perm.device)
        zeros = torch.zeros_like(ones)

        disc_loss = 0.5 * (cross_entropy(log_ratio_z, zeros) +
                           cross_entropy(log_ratio_z_perm, ones))

        self.optim.zero_grad()
        disc_loss.backward()
        self.optim.step()


class WAEGAN(AELoss):
    """
    Class that implements the adversarial version of the Wasserstein loss
    as found in "Wasserstein Autoencoders" Tolstikhin et al., 2019
    [https://arxiv.org/pdf/1711.01558.pdf].

    This version uses a trained discriminator to distinguish prior samples
    from posterior samples. The implementation is similar to FactorVAE,
    using a feedforward classifier. This model and the autoencoder are
    trained with conjugate gradient descent.
    """
    def __init__(self, reconstruction_loss='mse', lambda1=10.0, lambda2=0.0,
                 prior_var=1.0, lmbda_schedule=None, disc_args=None,
                 optim_kwargs=None):
        super().__init__(reconstruction_loss)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_var = prior_var
        self.lmbda_schedule = lmbda_schedule
        self.anneal = 1.0

        # if disc_args is None:
        #     disc_args = [('linear', [1000]), ('relu',)] * 6

        default_optim_kwargs = {'optimizer': 'adam', 'lr': 1e-3,
                                'betas': (0.5, 0.9)}
        if optim_kwargs is not None:
            default_optim_kwargs.update(optim_kwargs)

        # disc_args.append(('linear', [2]))
        # self.disc = feedforward.FeedForward(*disc_args)
        self.disc = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

        self.optim = init_optimizer(params=self.disc.parameters(),
                                    **default_optim_kwargs)
        self._batch_samples = None

    @property
    def disc_device(self):
        return next(self.disc.parameters()).device

    def train(self, mode=True):
        self.disc.train(mode)
        for p in self.disc.parameters():
            p.requires_grad = mode

    def eval(self):
        self.train(False)

    def _set_device(self, input_device):
        if self.disc_device is None or (self.disc_device != input_device):
            self.disc.to(device=input_device)

    def latent_term(self, z, z_params):
        # Hack to set the device
        self._set_device(z.device)
        self.eval()

        self._batch_samples = z.detach()

        log_z_ratio = self.disc(z)
        adv_term = (log_z_ratio[:, 0] - log_z_ratio[:, 1]).mean()

        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lmbda_schedule is not None:
            steps, min_anneal = self.lmbda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)

        if self._batch_samples is None:
            return

        # Train discriminator
        self.train()
        self.optim.zero_grad()

        z, self._batch_samples = self._batch_samples, None

        z_prior = self.prior_var * torch.randn_like(z)

        log_ratio_z = self.disc(z)
        log_ratio_z_prior = self.disc(z_prior)

        ones = z_prior.new_ones(z_prior.size(0), dtype=torch.long)
        zeros = torch.zeros_like(ones)

        disc_loss = 0.5 * (cross_entropy(log_ratio_z, zeros) +
                           cross_entropy(log_ratio_z_prior, ones))

        disc_loss.backward()
        self.optim.step()


class WAEMMD(AELoss):
    """
    Class that implements the Minimum Mean Discrepancy term in the latent space
    as found in "Wasserstein Autoencoders", Tolstikhin et al., (2019)
    [https://arxiv.org/pdf/1711.01558.pdf], with the modifications proposed in
    "Learning disentangled representations with Wasserstein Autoencoders"
    Rubenstein et al., 2018 [https://openreview.net/pdf?id=Hy79-UJPM].

    Unlike the adversarial version, this one relies on kernels to determine the
    distance between the distributions. While we allow any kernel, the default
    is the sum of inverse multiquadratics which has heavier tails than RBF. We
    also add an L1 penalty on the log-variance to prevent the encoders from
    becoming deterministic.
    """
    def __init__(self, reconstruction_loss='mse', lambda1=10, lambda2=1.0,
                 prior_type='norm', prior_var=1.0, kernel=None,
                 lmbda_schedule=None):

        super().__init__(reconstruction_loss)

        if kernel is None:
            kernel = partial(inv_multiquad_sum,
                             base_scale=10.0,
                             scales=torch.tensor([0.1, 0.2, 0.5, 1.0,
                                                  2.0, 5.0, 10.0]))

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_type = prior_type
        self.prior_var = prior_var
        self.lmbda_schedule = lmbda_schedule
        self.kernel = kernel
        self.anneal = 1.0
        # Save the indices of the combinations for reuse
        self._idxs = None

    def latent_term(self, z, z_params):
        if self.prior_type == 'norm':
            z_prior = self.prior_var * torch.randn_like(z)
        elif self.prior_type == 'unif':
            z_prior = self.prior_var * torch.rand_like(z) - 0.5
        else:
            raise ValueError('Unrecognized prior {}'.format(self.prior_type))

        if self._idxs is None or len(self._idxs[1]) != z.size(0) ** 2:
            self._idxs = mmd_idxs(z.size(0))

        adv_term = min_mean_discrepancy(z, z_prior, self.kernel, self._idxs)

        # L1 regularization of log-variance
        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lmbda_schedule is not None:
            steps, min_anneal = self.lmbda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)


# class tcWAEMMD(WAEMMD):
#     def __init__(self, reconstruction_loss='mse', lmbda=10, gamma=100,
#                  prior_var=1.0, kernel=None, lmbda_schedule=None):

#         super().__init__(reconstruction_loss, lmbda, prior_var, kernel,
#                         lmbda_schedule)
#         self.gamma = gamma

#     def latent_term(self, z, z_params):
#         z, z_marginal = z.chunk(2, 0)
#         z_marginal = permute_dims(z_marginal.detach())

#         reg_term = super().latent_term(z, z_params)
#         tc_term = min_mean_discrepancy(z, z_marginal, self.kernel, self._idxs)

#         return reg_term + self.gamma * tc_term


# class SlowVAE(AELoss):
#     def __init__(self, reconstruction_loss='bce', rate_prior=6.0, gamma=6.0):
#         super().__init__(reconstruction_loss)
#         self.rate_prior = rate_prior
#         self.gamma = gamma

#     def latent_term(self, z_sample, z_params):
#         z_orginal, z_transform = z_sample.chunk(2, dim=0)

#         z_mean, z_logvar = z_params
#         z_og_mu, z_transf_mu = z_mean.chunk(2, dim=0)
#         z_og_logvar, z_transf_logvar = z_logvar.chunk(2, dim=0)

#         # TODO: something something cross entropy
#         pass


class LieVAELoss(CCIVAE):
    def __init__(self, reconstruction_loss='bce', hy_rec=0.1, hy_hes=40,
                 hy_commute=0, subspace_sizes=None, subgroup_sizes=None,
                 gamma=100.0, capacity=0.0, capacity_schedule=None):
        super().__init__(reconstruction_loss, gamma, capacity, capacity_schedule)

        self.hy_hes = hy_hes
        self.hy_rec = hy_rec
        self.hy_commute = hy_commute
        self.subspace_sizes = subspace_sizes
        self.subgroup_sizes = subgroup_sizes

    def latent_term(self, z_sample, z_params):
        (ge, _, gd) = z_sample
        (mu, logvar, lie_basis) = z_params

        kl_term = self.kl_term(mu, logvar)
        group_loss = self.group_loss(ge, gd, lie_basis)
        return group_loss + kl_term

    def kl_term(self, mu, logvar):
        kl_div = gauss2standard_kl(mu, logvar).sum() / len(mu)
        return self.gamma * (kl_div - self.capacity).abs()

    def group_loss(self, group_feats_E, group_feats_G, lie_alg_basis_ls):
        b_idx = 0
        hessian_loss = 0.
        commute_loss = 0.

        for i, subspace_size in enumerate(self.subspace_sizes):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(self.subgroup_sizes[i]))

                assert list(lie_alg_basis_ls[b_idx].size())[-1] == mat_dim

                lie_alg_basis_mul_ij = calc_basis_mul_ij(
                    lie_alg_basis_ls[b_idx:e_idx])  # XY
                hessian_loss += calc_hessian_loss(lie_alg_basis_mul_ij)
                if self.hy_commute > 0:
                    commute_loss += calc_commute_loss(lie_alg_basis_mul_ij)

            b_idx = e_idx

        rec_loss = torch.mean(torch.sum(
            torch.square(group_feats_E - group_feats_G), dim=1))

        rec_loss *= self.hy_rec
        hessian_loss *= self.hy_hes
        commute_loss *= self.hy_commute

        return hessian_loss + commute_loss + rec_loss


# Metrics
class ReconstructionNLL(_Loss):
    """
    Standard reconstruction of images. There are two options, minimize the
    Bernoulli loss (i.e. per pixel binary cross entropy) or MSE (i.e. Gaussian
    likelihoid).
    """
    def __init__(self, loss='bce'):
        super().__init__(reduction='batchmean')
        if loss == 'bce':
            recons_loss = logits_bce
        elif loss == 'mse':
            recons_loss = mse_loss
        elif not callable(loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(loss))
        self.loss = recons_loss

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            recons = input[0]
        else:
            recons = input

        return self.loss(recons, target, reduction='sum') / target.size(0)


class GaussianKLDivergence(_Loss):
    """
    Computes the KL divergence between a latent variable and a standard normal
    distribution. Optinally, allows for computing the KL for a single
    dimension. This can be used to see which units are being used by the model
    to solve the task.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input, targets):
        _, _, (mu, logvar) = input

        if self.dim >= 0:
            mu, logvar = mu[:, self.dim], logvar[:, self.dim]

        kl = gauss2standard_kl(mu, logvar).sum()
        return kl / targets.size(0)


class MomentMatching(_Loss):
    def __init__(self, empirical=True):
        super().__init__()
        self.empirical = empirical

    def forward(self, input, targets):
        _, z, params = input

        if self.empirical:
            M = z - z.mean(axis=0, keepdims=True)
        else:
            mu = params[0]
            M = mu - mu.mean(axis=0, keepdims=True)

        cov_sqrd = (M.T @ M / (z.size(0) - 1)) ** 2
        weights = 1.0 - torch.diag(torch.ones_like(cov_sqrd[0]))

        return (cov_sqrd * weights).sum()
