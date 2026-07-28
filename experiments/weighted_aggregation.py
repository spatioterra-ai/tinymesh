"""Show weighted CSR forward and both first-order gradients."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor

from tinymesh import Graph


@dataclass(frozen=True)
class Observation:
    device: str
    output: list[list[float]]
    node_gradient: list[list[float]]
    edge_weight_gradient: list[float]


def observe(device: str) -> Observation:
    graph = Graph(3, source=[0, 1, 0], target=[2, 2, 1])
    values = Tensor([[2.0], [4.0], [8.0]], device=device).realize()
    edge_weight = Tensor([2.0, -1.0, 3.0], device=device).realize()
    output = graph.sum(values, edge_weight=edge_weight)
    node_gradient, edge_weight_gradient = output.gradient(
        values,
        edge_weight,
        gradient=Tensor([[0.0], [5.0], [7.0]], device=device),
    )
    Tensor.realize(output, node_gradient, edge_weight_gradient)
    return Observation(
        device,
        output.tolist(),
        node_gradient.tolist(),
        edge_weight_gradient.tolist(),
    )


def main() -> None:
    print(json.dumps(asdict(observe(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
