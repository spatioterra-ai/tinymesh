"""Compose bidirectional diffusion inside a gated recurrent cell."""

from __future__ import annotations

from tinygrad import Tensor, nn

from experiments.directed_diffusion import DirectedDiffusion


class DiffusionGRU:
    def __init__(self, in_features: int, hidden_features: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        width = 3 * (in_features + hidden_features)
        self.gates = nn.Linear(width, 2 * hidden_features)
        self.candidate = nn.Linear(width, hidden_features)
        self.in_features = in_features
        self.hidden_features = hidden_features

    def __call__(
        self,
        values: Tensor,
        diffusion: DirectedDiffusion,
        hidden: Tensor | None = None,
    ) -> Tensor:
        expected = (diffusion.graph.nodes, self.in_features)
        if values.ndim < 2 or values.shape[-2:] != expected:
            raise ValueError(
                f"values must have shape [..., {diffusion.graph.nodes}, "
                f"{self.in_features}], got {values.shape}"
            )
        hidden = self._hidden(values, hidden)
        gates = self.gates(self._basis(values.cat(hidden, dim=-1), diffusion))
        update = gates[..., :self.hidden_features].sigmoid()
        reset = gates[..., self.hidden_features:].sigmoid()
        candidate = self.candidate(
            self._basis(values.cat(hidden * reset, dim=-1), diffusion)
        ).tanh()
        return update * hidden + (1 - update) * candidate

    def _basis(self, values: Tensor, diffusion: DirectedDiffusion) -> Tensor:
        forward, reverse = diffusion(values)
        return values.cat(forward, reverse, dim=-1)

    def _hidden(self, values: Tensor, hidden: Tensor | None) -> Tensor:
        if hidden is None:
            return Tensor.zeros(
                *values.shape[:-1],
                self.hidden_features,
                dtype=values.dtype,
                device=values.device,
            )
        expected = (*values.shape[:-1], self.hidden_features)
        if hidden.shape != expected:
            raise ValueError(f"hidden must have shape {expected}, got {hidden.shape}")
        if hidden.dtype != values.dtype or hidden.device != values.device:
            raise ValueError("hidden and values must share dtype and device")
        return hidden


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
