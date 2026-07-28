"""Inspect the pinned PyG Temporal chickenpox signal."""

import json
from dataclasses import asdict, dataclass

from tinygrad import Device

from tinymesh.datasets import chickenpox


@dataclass(frozen=True)
class Observation:
    device: str
    nodes: int
    edges: int
    self_loops: int
    snapshots: int
    train_snapshots: int
    test_snapshots: int
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]


def observe(device: str) -> Observation:
    dataset = chickenpox(device=device)
    train, test = dataset.split(0.8)
    x, y = dataset[0]
    return Observation(
        device,
        nodes=dataset.graph.nodes,
        edges=dataset.graph.edges,
        self_loops=sum(source == target for source, target in zip(dataset.graph.source, dataset.graph.target)),
        snapshots=len(dataset),
        train_snapshots=len(train),
        test_snapshots=len(test),
        x_shape=x.shape,
        y_shape=y.shape,
    )


def main() -> None:
    print(json.dumps(asdict(observe(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
