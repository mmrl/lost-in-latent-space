"""
Module with stochastic layers

Currently only the diagonal gaussian and a couple of variants I was testing
exist here. Original idea was to add more. This just take a mean and a
logvariance and performs the reparameterization trick on the result.

Appart from the standard one, there is a variant with input-independant but
learned logar, wich means that all inputs share the same covariance. This is
similar to the idea behind LDA.
"""

import torch
import torch.nn as nn


class StochasticLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def reparam(self, *params):
        raise NotImplementedError()

    def sample(self, inputs=None, nsamples=1):
        raise NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()


# Gaussian stochastic layers
class DiagonalGaussian(StochasticLayer):
    def __init__(self, latent_size, std_fn='exp'):
        super().__init__()
        self.size = latent_size
        self.std_fn = std_fn

    def reparam(self, mu, logvar, random_eval=False):
        if self.training or random_eval:
            # std = exp(log(var))^0.5
            if self.std_fn == 'exp':
                std = logvar.mul(0.5).exp()
            elif self.std_fn == 'softplus':
                std = nn.functional.softplus(logvar)
            else:
                raise ValueError()
            eps = torch.randn_like(std)
            # z = mu + std * eps
            return mu.addcmul(std, eps)
        return mu

    def sample(self, inputs, n_samples=1):
        inputs = inputs.unsqueeze_(1).expand(-1, n_samples, -1)
        mu, logvar = inputs.chunk(2, dim=-1)

        return self.reparam(mu, logvar, random_eval=True)

    def forward(self, inputs):
        mu, logvar = inputs.chunk(2, dim=-1)
        return self.reparam(mu, logvar), (mu, logvar)

    def extra_repr(self):
        return 'size={}'.format(self.size)


class BoundedGaussian(DiagonalGaussian):
    def reparam(self, mu, logvar, random_eval=False):
        mu = torch.tanh(mu)
        return super().reparam(mu, logvar, random_eval)


class HomoscedasticGaussian(DiagonalGaussian):
    def __init__(self, latent_size):
        super().__init__(latent_size)
        self._logvar_z = nn.parameter.Parameter(torch.zeros(latent_size,
                                                dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self._logvar_z)

    def forward(self, inputs):
        return self.reparam(inputs, self.logvar_z), (inputs, self.logvar_z)


# Uniform stochastic layers

class Uniform(StochasticLayer):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

    def reparam(self, mu, loglen, random_eval=True):
        if self.training or random_eval:
            length = loglen.exp()
            a = mu - length
            b = mu + length
            u = torch.rand_like(a)
            return a.addcmul(u, (b - a))
        return mu

    def sample(self, inputs, n_samples=1):
        inputs = inputs.unsqueeze_(1).expand(-1, n_samples, -1)
        mu, loglen = inputs.chunk(2, dim=-1)

        return self.reparam(mu, loglen, random_eval=True)

    def forward(self, inputs):
        mu, loglen = inputs.chunk(2, dim=-1)
        return self.reparam(mu, loglen, random_eval=True), (mu, log_length)


class BoundedUniform(Uniform):
    def reparam(self, mu, loglen, random_eval=False):
        mu = torch.tanh(mu)
        return super().reparam(mu, loglen, random_eval)
