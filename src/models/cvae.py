"""
Architecture used for the CascadeVAE model, using discrete latents.

Code was translated to PyTorch from the original TensorFlow repository found here:
    https://github.com/snu-mllab/DisentanglementICML19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .stochastic import DiagonalGaussian
from .initialization import weights_init
from .matching import SolveMaxMatching


logits_bce = F.binary_cross_entropy_with_logits


class CascadeVAE(nn.Module):
    """
    VAE with discrete and continuous latente variables
    """
    def __init__(self, latente_size: int, n_cat: int, encoder: nn.Module,
            decoder: nn.Module, disc_warmup: int=-1, lmbda: float=0.001) -> None:
        super().__init__()

        assert n_cat > 1

        self.encoder = encoder
        self.latent = DiagonalGaussian(latente_size, std_fn='exp')
        self.decoder = decoder

        self.n_cat = n_cat
        self.register_buffer('_train_steps', torch.tensor(0, dtype=torch.int64))
        self.warmup_steps = disc_warmup
        self.mcf = lambda B: SolveMaxMatching(nworkers=B, ntasks=n_cat, k=1,
                                              pairwise_lamb=lmbda)

    def add_onehot(self, z):
        """
        Create continuous + discrete latent representations
        """
        B = len(z)

        z = z.unsqueeze(1).expand(-1, self.n_cat, -1)
        d_latent = F.one_hot(torch.arange(0, self.n_cat))

        e = d_latent[None].expand(B, -1, -1).to(device=z.device, dtype=z.dtype)

        return torch.cat([z, e], dim=2)

    def compute_discrete(self, inputs, z_cont):
        B = inputs.shape[0]

        if self.training and self._train_steps < self.warmup_steps:
            z_disc = inputs.new_zeros((B, self.n_cat))
            return z_disc, z_disc

        z_all = self.add_onehot(z_cont)

        recons = self.decoder(z_all.flatten(end_dim=1))
        inputs = torch.repeat_interleave(inputs, self.n_cat, dim=0)

        image_dims = list(range(1, len(recons.shape)))
        nll = logits_bce(recons, inputs, reduction='none').sum(dim=image_dims)
        nll = nll.unflatten(0, (B, self.n_cat))

        if self.training:
            cost_matrix = -nll.cpu().detach_().numpy()
            z_disc = nll.new_tensor(self.mcf(B).solve(cost_matrix)[1])
        else:
            idx = torch.argmin(nll, dim=-1)
            z_disc = nll.new_zeros(B, self.n_cat)
            z_disc[list(range(B)), idx] = 1.0

        return z_disc, nll

    def forward(self, inputs):
        if self.training:
            self._train_steps = self._train_steps + 1

        h = self.encoder(inputs)

        z, z_params = self.latent(h)  # z_params = (mu, logvar)
        z_disc, nll = self.compute_discrete(inputs, z)

        z_params = z_params, nll

        z_all = torch.cat([z, z_disc], dim=-1)
        recons = self.decoder(z_all)

        return recons, z_all, z_params
