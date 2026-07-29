"""Prove sparse source-normalized diffusion in both graph directions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor

from tinymesh import Graph
from tinymesh.nn import DirectedDiffusion


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
