"""Prove fixed-graph temporal recurrence over existing Graph operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph
from tinymesh.nn import GCNConv


class TGCN:
    def __init__(self, in_features: int, hidden_features: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        self.graph_projection = GCNConv(in_features, 3 * hidden_features, bias=False)
        self.update = nn.Linear(2 * hidden_features, hidden_features)
        self.reset = nn.Linear(2 * hidden_features, hidden_features)
        self.candidate = nn.Linear(2 * hidden_features, hidden_features)
        self.hidden_features = hidden_features

    def __call__(self, values: Tensor, graph: Graph, hidden: Tensor | None = None) -> Tensor:
        graph_state = self.graph_projection(values, graph)
        update_input = graph_state[..., :self.hidden_features]
        reset_input = graph_state[..., self.hidden_features:2 * self.hidden_features]
        candidate_input = graph_state[..., 2 * self.hidden_features:]
        hidden = self._hidden(update_input, hidden)
        update = self.update(update_input.cat(hidden, dim=-1)).sigmoid()
        reset = self.reset(reset_input.cat(hidden, dim=-1)).sigmoid()
        candidate = self.candidate(candidate_input.cat(hidden * reset, dim=-1)).tanh()
        return update * hidden + (1 - update) * candidate

    def _hidden(self, reference: Tensor, hidden: Tensor | None) -> Tensor:
        if hidden is None:
            return Tensor.zeros(
                *reference.shape[:-1],
                self.hidden_features,
                dtype=reference.dtype,
                device=reference.device,
            )
        expected = (*reference.shape[:-1], self.hidden_features)
        if hidden.shape != expected:
            raise ValueError(f"hidden must have shape {expected}, got {hidden.shape}")
        if hidden.dtype != reference.dtype or hidden.device != reference.device:
            raise ValueError("hidden and values must share dtype and device")
        return hidden


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    candidate_gradient: float
    final_loss: float
    candidate_weight: float


def fit_one_step(device: str) -> Observation:
    graph = Graph(2, [0, 1, 0], [0, 1, 1])
    snapshots = (
        Tensor([[1.0], [0.0]], device=device).realize(),
        Tensor.zeros(2, 1, device=device).realize(),
    )
    target = Tensor([[1.0]], device=device).realize()
    model = TGCN(1, 1)
    _set_witness_parameters(model, device)
    optimizer = nn.optim.SGD([model.graph_projection.linear.weight], lr=1.0, fused=False)

    def loss() -> Tensor:
        hidden = model(snapshots[0], graph)
        for snapshot in snapshots[1:]:
            hidden = model(snapshot, graph, hidden)
        return (hidden[1:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        gradient = model.graph_projection.linear.weight.grad
        if gradient is None:
            raise RuntimeError("candidate graph parameter has no gradient")
        candidate_gradient = gradient[2, 0].item()
        training_loss.realize(*optimizer.schedule_step())

    return Observation(
        device,
        initial_loss,
        candidate_gradient,
        loss().item(),
        model.graph_projection.linear.weight[2, 0].item(),
    )


def _set_witness_parameters(model: TGCN, device: str) -> None:
    model.graph_projection.linear.weight = Tensor([[0.0], [0.0], [1.0]], device=device).realize()
    for gate in (model.update, model.reset):
        gate.weight = Tensor.zeros(1, 2, device=device).realize()
        gate.bias = Tensor.zeros(1, device=device).realize()
    model.candidate.weight = Tensor([[1.0, 0.0]], device=device).realize()
    model.candidate.bias = Tensor.zeros(1, device=device).realize()


def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
