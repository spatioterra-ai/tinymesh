"""Prove a trainable mean-GraphSAGE caller over CSR aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph


class MeanSAGE:
    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        self.root = nn.Linear(in_features, out_features, bias=False)
        self.neighbor = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        messages = self.neighbor(values)
        if not isinstance(messages.device, str):
            raise ValueError("mean GraphSAGE requires one device")
        degree = graph.in_degree(device=messages.device)
        inverse_degree = degree.maximum(1).cast(messages.dtype).reciprocal().reshape(-1, 1)
        return self.root(values) + graph.sum(messages) * inverse_degree


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    neighbor_gradient: float
    final_loss: float
    root_weight: float
    neighbor_weight: float


def fit_one_step(device: str) -> Observation:
    graph = Graph(4, [0, 1], [2, 3])
    values = Tensor([[1.0], [-1.0], [0.0], [0.0]], device=device).realize()
    target = Tensor([[1.0], [-1.0]], device=device).realize()
    model = MeanSAGE(1, 1)
    model.root.weight = Tensor.zeros(1, 1, device=device).realize()
    model.neighbor.weight = Tensor.zeros(1, 1, device=device).realize()
    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=0.5, fused=False)

    def loss() -> Tensor:
        return (model(values, graph)[2:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        gradient = model.neighbor.weight.grad
        if gradient is None:
            raise RuntimeError("neighbor parameter has no gradient")
        neighbor_gradient = gradient.item()
        training_loss.realize(*optimizer.schedule_step())

    return Observation(
        device,
        initial_loss,
        neighbor_gradient,
        loss().item(),
        model.root.weight.item(),
        model.neighbor.weight.item(),
    )


def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
