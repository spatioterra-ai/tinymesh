"""Prove one trainable graph-attention caller over sparse Graph primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph


class GAT:
    def __init__(self, in_features: int, out_features: int, negative_slope: float = 0.2) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")
        self.linear = nn.Linear(in_features, out_features, bias=False)
        bound = 1 / sqrt(out_features)
        self.source_attention = Tensor.uniform(out_features, low=-bound, high=bound)
        self.target_attention = Tensor.uniform(out_features, low=-bound, high=bound)
        self.negative_slope = negative_slope

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        state = self.linear(values)
        source_score = (state * self.source_attention).sum(axis=1, keepdim=True)
        target_score = (state * self.target_attention).sum(axis=1, keepdim=True)
        edge_score = (
            graph.edge_values(source_score, endpoint="source")
            + graph.edge_values(target_score, endpoint="target")
        ).reshape(-1)
        attention = graph.softmax(edge_score.leaky_relu(self.negative_slope))
        return graph.sum(state, attention)


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    source_attention_gradient: float
    final_loss: float
    linear_weight: float
    source_attention: float
    target_attention: float


def fit_one_step(device: str) -> Observation:
    graph = Graph(3, [0, 1], [2, 2])
    values = Tensor([[1.0], [-1.0], [0.0]], device=device).realize()
    target = Tensor([[1.0]], device=device).realize()
    model = GAT(1, 1)
    model.linear.weight = Tensor([[1.0]], device=device).realize()
    model.source_attention = Tensor([1.0], device=device).realize()
    model.target_attention = Tensor([0.0], device=device).realize()
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
        source_attention_gradient = gradient.item()
        training_loss.realize(*optimizer.schedule_step())

    return Observation(
        device,
        initial_loss,
        source_attention_gradient,
        loss().item(),
        model.linear.weight.item(),
        model.source_attention.item(),
        model.target_attention.item(),
    )


def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
