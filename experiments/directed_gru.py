"""Compose bidirectional diffusion inside a gated recurrent cell."""

from __future__ import annotations

from tinygrad import Tensor, nn

from tinymesh.nn import DiffusionGRU, DirectedDiffusion


class DiffusionForecast:
    def __init__(self, in_features: int, hidden_features: int) -> None:
        self.cell = DiffusionGRU(in_features, hidden_features)
        self.readout = nn.Linear(hidden_features, 1)

    def __call__(
        self,
        values: Tensor,
        diffusion: DirectedDiffusion,
        *,
        realize_steps: bool = False,
    ) -> Tensor:
        hidden = None
        for step in range(values.shape[1]):
            hidden = self.cell(values[:, step], diffusion, hidden)
            if realize_steps:
                hidden.realize()
        return self.readout(hidden)
