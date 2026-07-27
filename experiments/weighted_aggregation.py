"""Show weighted CSR forward and both first-order gradients."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor

from experiments.csr_aggregation import CSRTopology, csr_edge_weighted_sum


@dataclass(frozen=True)
class Observation:
    device: str
    csr_edge_order: tuple[int, ...]
    output: list[list[float]]
    node_gradient: list[list[float]]
    edge_weight_gradient: list[float]


def observe(device: str) -> Observation:
    topology = CSRTopology(3, source=[0, 1, 0], target=[2, 2, 1])
    values = Tensor([[2.0], [4.0], [8.0]], device=device).realize()
    edge_weight = Tensor(topology.lower_edge_values([2.0, -1.0, 3.0]), device=device).realize()
    output = csr_edge_weighted_sum(values, topology, edge_weight)
    node_gradient, edge_weight_gradient = output.gradient(
        values,
        edge_weight,
        gradient=Tensor([[0.0], [5.0], [7.0]], device=device),
    )
    Tensor.realize(output, node_gradient, edge_weight_gradient)
    return Observation(
        device,
        topology.edge_order,
        output.tolist(),
        node_gradient.tolist(),
        edge_weight_gradient.tolist(),
    )


def main() -> None:
    print(json.dumps(asdict(observe(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
