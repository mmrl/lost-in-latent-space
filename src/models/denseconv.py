import warnings

import torch
import torch.nn as nn


class DenseConv(nn.Module):
    def __init__(self, composite_functions, n_layers=4):
        super().__init__()
        if n_layers == 1:
            warnings.warn("Only one layer used in DenseConv,"
                          "use standard convolution instead",
                          warnings.WarningMessage, stacklevel=2)

        self.n_layers = n_layers
        self.composite_function = nn.ModuleList(*composite_functions)

    def forward(self, inputs):
        y = self.composite_function(inputs)

        for _ in range(self.n_layers - 1):
            inputs = torch.cat([inputs, y], dim=1)
            y = self.composite_function(inputs)

        return y
