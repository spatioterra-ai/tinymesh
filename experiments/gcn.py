"""Prove a trainable GCN caller over CSR aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph
from tinymesh.nn import GCNConv


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    weight_gradient: float
    final_loss: float
    weight: float


def fit_one_step(device: str) -> Observation:
    graph = Graph(
        4,
        [0, 1, 2, 3, 0, 2, 1, 3],
        [0, 1, 2, 3, 2, 0, 3, 1],
    )
    values = Tensor([[1.0], [-1.0], [0.0], [0.0]], device=device).realize()
    target = Tensor([[1.0], [-1.0]], device=device).realize()
    model = GCNConv(1, 1, bias=False)
    model.linear.weight = Tensor.zeros(1, 1, device=device).realize()
    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=2.0, fused=False)

    def loss() -> Tensor:
        return (model(values, graph)[2:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        gradient = model.linear.weight.grad
        if gradient is None:
            raise RuntimeError("GCN parameter has no gradient")
        weight_gradient = gradient.item()
        training_loss.realize(*optimizer.schedule_step())

    return Observation(
        device,
        initial_loss,
        weight_gradient,
        loss().item(),
        model.linear.weight.item(),
    )

def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
