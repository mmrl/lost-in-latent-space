import torch
import torch.nn as nn


class SpatialBroadcast(nn.Module):
    def __init__(self, height, width=None) -> None:
        super().__init__()
        if width is  None:
            width = height

        self.width = width
        self.height = height

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tiled = torch.tile(z[(..., None, None)], (self.height, self.width))
        # tiled = tiled.permute(0, 2, 3, 1)

        x = torch.linspace(-1, 1, self.width, device=z.device)
        y = torch.linspace(-1, 1, self.height, device=z.device)

        # use 'xy' indexing to swap the cardinality of the resulting dimensions
        x, y = torch.meshgrid(x, y, indexing='xy')

        # expand 'x' and 'y' along the non-broadcast dimensions to match 'tiled'
        x = x[None, None].expand(len(z), -1, -1, -1)
        y = y[None, None].expand(len(z), -1, -1, -1)

        return torch.cat([tiled, x, y], dim=1).contiguous()

    def __repr__(self):
        return 'SpatialBroadcast({},{})'.format(self.height, self.width)
