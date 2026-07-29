"""Prove metric graph geometry by composing existing Tinymesh primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor

from tinymesh import Graph


def radial_message(
    graph: Graph,
    position: Tensor,
    values: Tensor,
    decay: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source = graph.edge_values(position, endpoint="source")
    target = graph.edge_values(position, endpoint="target")
    displacement = target - source
    distance = (displacement * displacement).sum(axis=-1).sqrt()
    weight = (-decay * distance).exp()
    return displacement, distance, weight, graph.sum(values, edge_weight=weight)


@dataclass(frozen=True)
class Observation:
    device: str
    displacement: list[list[float]]
    distance: list[float]
    weight: list[float]
    output: list[list[float]]
    position_gradient: list[list[float]]
    value_gradient: list[list[float]]
    weight_gradient: list[float]
    decay_gradient: float


def observe(device: str) -> Observation:
    graph = Graph(4, source=[0, 1, 0, 2], target=[2, 2, 1, 3])
    position = Tensor(
        [[0.0, 0.0], [0.0, 3.0], [4.0, 0.0], [4.0, 3.0]],
        device=device,
    ).realize()
    values = Tensor(
        [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0], [5.0, 2.0]],
        device=device,
    ).realize()
    decay = Tensor(0.25, device=device).realize()
    displacement, distance, weight, output = radial_message(
        graph,
        position,
        values,
        decay,
    )
    position_gradient, value_gradient, weight_gradient, decay_gradient = output.gradient(
        position,
        values,
        weight,
        decay,
        gradient=Tensor(
            [[0.0, 0.0], [1.0, -2.0], [3.0, 1.0], [-1.0, 2.0]],
            device=device,
        ),
    )
    Tensor.realize(
        displacement,
        distance,
        weight,
        output,
        position_gradient,
        value_gradient,
        weight_gradient,
        decay_gradient,
    )
    return Observation(
        device,
        displacement.tolist(),
        distance.tolist(),
        weight.tolist(),
        output.tolist(),
        position_gradient.tolist(),
        value_gradient.tolist(),
        weight_gradient.tolist(),
        decay_gradient.item(),
    )


def main() -> None:
    print(json.dumps(asdict(observe(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
