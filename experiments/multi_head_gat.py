"""Prove independent attention heads compose over scalar Graph operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from tinymesh.nn import GATConv
from tinymesh import Graph


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    source_attention_gradient: tuple[float, float]
    final_loss: float
    linear_weight: tuple[float, float]
    source_attention: tuple[float, float]


def fit_one_step(device: str) -> Observation:
    graph = Graph(3, [0, 1], [2, 2])
    values = Tensor([[1.0], [-1.0], [0.0]], device=device).realize()
    target = Tensor([[1.0, -1.0]], device=device).realize()
    model = GATConv(1, 1, heads=2, bias=False)
    model.linear.weight = Tensor([[1.0], [1.0]], device=device).realize()
    model.source_attention = Tensor([[1.0], [-1.0]], device=device).realize()
    model.target_attention = Tensor.zeros(2, 1, device=device).realize()
    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=0.1, fused=False)

    def loss() -> Tensor:
        return (model(values, graph)[2:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        gradient = model.source_attention.grad
        if gradient is None:
            raise RuntimeError("source attention has no gradient")
        source_attention_gradient = gradient.flatten().tolist()
        training_loss.realize(*optimizer.schedule_step())

    linear_weight = model.linear.weight.flatten().tolist()
    source_attention = model.source_attention.flatten().tolist()
    return Observation(
        device,
        initial_loss,
        (source_attention_gradient[0], source_attention_gradient[1]),
        loss().item(),
        (linear_weight[0], linear_weight[1]),
        (source_attention[0], source_attention[1]),
    )


def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
