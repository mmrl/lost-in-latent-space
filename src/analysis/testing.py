"""
Utility functions for getting reconstructions and scores from the models.
"""

import torch
import pandas as pd
from ignite.engine import create_supervised_evaluator
from torch.utils.data.dataloader import DataLoader


def get_recons(model, data, loss='bce'):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        recons, targets = [], []

        for x, y in data:
            x = x.to(device=device)
            r = model(x)

            if isinstance(r, tuple):
                r = r[0]

            if loss == 'bce':
                r = r.sigmoid()
            else:
                r = r.clamp(0, 1)

            recons.append(r.cpu())
            targets.append(y)

    recons = torch.cat(recons)
    targets = torch.cat(targets)

    return recons, targets


def model_scores(models, data, model_names, metric, device):
    scores = []
    loader = DataLoader(data, batch_size=120, num_workers=8, pin_memory=True)
    for m in models:
        engine = create_supervised_evaluator(m, metric, device)
        metrics = engine.run(loader).metrics

        index = pd.Index(metrics.keys(), name='Metric')
        metric_scores = pd.Series(metrics.values(), index=index, name='value')

        scores.append(metric_scores)

    return pd.concat(scores, keys=model_names, names=['Model'])


def infer(model, data, **kwargs):
    dataloader_args = {'batch_size': 120, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in loader:
            x = x.to(device=device)
            z = model(x)[1]

            if isinstance(z, tuple):
                z = z[1]

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets
