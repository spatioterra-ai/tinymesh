"""Prove sparse source-normalized diffusion in both graph directions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor, dtypes

from tinymesh import Graph


class DirectedDiffusion:
    """A fixed directed graph with positive, source-normalized edge affinity."""

    def __init__(self, graph: Graph, affinity: Tensor) -> None:
        if affinity.ndim != 1 or affinity.shape[0] != graph.edges:
            raise ValueError(f"affinity must have shape [{graph.edges}], got {affinity.shape}")
        if not dtypes.is_float(affinity.dtype):
            raise ValueError(f"affinity must have a floating dtype, got {affinity.dtype}")
        if not isinstance(affinity.device, str):
            raise ValueError("directed diffusion requires one device")

        self.graph = graph
        self.reverse = Graph(graph.nodes, graph.target, graph.source)
        one = Tensor.ones(graph.nodes, 1, dtype=affinity.dtype, device=affinity.device)
        outgoing = self.reverse.sum(one, edge_weight=affinity)
        incoming = graph.sum(one, edge_weight=affinity)
        self.forward_weight = affinity / graph.edge_values(
            outgoing,
            endpoint="source",
        ).flatten()
        self.reverse_weight = affinity / graph.edge_values(
            incoming,
            endpoint="target",
        ).flatten()

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.graph.sum(values, edge_weight=self.forward_weight),
            self.reverse.sum(values, edge_weight=self.reverse_weight),
        )


@dataclass(frozen=True)
class Observation:
    device: str
    forward_weight: list[float]
    reverse_weight: list[float]
    forward: list[list[float]]
    reverse: list[list[float]]
    value_gradient: list[list[float]]
    affinity_gradient: list[float]


def observe(device: str) -> Observation:
    graph = Graph(
        5,
        source=[0, 0, 1, 2, 2, 2],
        target=[1, 2, 2, 0, 1, 1],
    )
    affinity = Tensor([1.0, 3.0, 2.0, 4.0, 2.0, 1.0], device=device).realize()
    values = Tensor(
        [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0], [5.0, 2.0], [7.0, -3.0]],
        device=device,
    ).realize()
    forward_gradient = Tensor(
        [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0], [2.0, 0.0], [5.0, -3.0]],
        device=device,
    )
    reverse_gradient = Tensor(
        [[-2.0, 1.0], [1.0, 3.0], [4.0, -1.0], [0.0, 2.0], [-3.0, 5.0]],
        device=device,
    )

    diffusion = DirectedDiffusion(graph, affinity)
    forward, reverse = diffusion(values)
    loss = (forward * forward_gradient).sum() + (reverse * reverse_gradient).sum()
    value_gradient, affinity_gradient = loss.gradient(values, affinity)
    Tensor.realize(forward, reverse, value_gradient, affinity_gradient)
    return Observation(
        device,
        diffusion.forward_weight.tolist(),
        diffusion.reverse_weight.tolist(),
        forward.tolist(),
        reverse.tolist(),
        value_gradient.tolist(),
        affinity_gradient.tolist(),
    )


def main() -> None:
    print(json.dumps(asdict(observe(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
