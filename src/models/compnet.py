import torch
import torch.nn as nn


class FixedInterpComp(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        assert latent_size >= n_actions
        self.latent_size = latent_size
        self.n_dummy = latent_size - n_actions

    def forward(self, z, actions):
        batch_size = len(actions)

        dummy_vars = actions.new_zeros((batch_size, self.n_dummy))
        actions = torch.cat([actions, dummy_vars], axis=1)

        z_ref, z_trans = z.reshape(-1, 2, self.latent_size).chunk(2, 1)
        z_ref, z_trans = z_ref.squeeze(1), z_trans.squeeze(1)

        return z_ref * (1 - actions) + z_trans * actions


class LinearComp(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.ref_proj = nn.Linear(n_actions + latent_size, latent_size)
        self.trans_proj = nn.Linear(n_actions + latent_size, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.reshape(-1, 2, self.latent_size).chunk(2, 1)

        z_ref = torch.cat([z_ref.squeeze(1), actions], dim=1).contiguous()
        z_trans = torch.cat([z_trans.squeeze(1), actions], dim=1).contiguous()

        return self.ref_proj(z_ref) + self.trans_proj(z_trans)


class InterpComp(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.linear = nn.Linear(2 * latent_size + n_actions, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.reshape(-1, 2, self.latent_size).chunk(2, 1)
        z_ref, z_trans = z_ref.squeeze(1), z_trans.squeeze(1)

        w = self.linear(torch.cat([z_ref, z_trans, actions], dim=1)).sigmoid()
        return z_ref * w + z_trans * (1.0 - w)


class MLPComp(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.n_actions = n_actions
        self.latent_size = latent_size

        input_size = 2 * latent_size + n_actions
        self.projection = nn.Sequential(nn.Linear(input_size, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, latent_size))

    def forward(self, z, actions):
        zpa = torch.cat([z.reshape(-1, self.latent_size), actions], dim=1)
        return self.projection(zpa.contiguous())


class CompositionNet(nn.Module):
    def __init__(self, vae, composition_op):
        super().__init__()
        self.vae = vae
        self.composition_op = composition_op

    @property
    def latent_size(self):
        return self.vae.latent_size

    @property
    def n_actions(self):
        return self.composition_op.n_actions

    def forward(self, inputs):
        inputs, actions = inputs
        input_size = inputs.shape

        # Format inputs so that we have shape (2 * batch_size, input_size)
        # and corresponding reference and transform images follow each other
        inputs = inputs.flatten(0, 1)

        # Compute latent values for the two images (reference and transform),
        # reshape the values so that corresponding pairs are in the same row
        recons, z, params = self.vae(inputs)
        # z, params = self.vae.latent(self.vae.encoder(inputs))

        # Transform the latents according to the action and decode
        z_transf = self.composition_op(z, actions)
        transformation = self.vae.decoder(z_transf)

        # reshape results
        transformation = transformation.reshape(-1, *input_size[-3:])
        recons = recons.reshape(*input_size)
        z = z.reshape(input_size[0], 2, -1)

        # Concatenate vectors
        recons = torch.cat([recons, transformation.unsqueeze(1)], dim=1)
        z = torch.cat([z, z_transf.unsqueeze(1)], dim=1)

        recons = recons.contiguous()
        z = z.contiguous()

        return recons, z, params
