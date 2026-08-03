"""Inspect the pinned MUTAG molecular graph collection."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Device

from tinymesh.datasets import mutag


@dataclass(frozen=True)
class Observation:
    device: str
    graphs: int
    classes: int
    class_counts: tuple[int, ...]
    nodes: int
    minimum_nodes: int
    maximum_nodes: int
    directed_edges: int
    bonds: int
    reciprocal_edges: int
    node_types: tuple[str, ...]
    node_type_counts: tuple[int, ...]
    bond_types: tuple[str, ...]
    bond_type_counts: tuple[int, ...]


def observe(path: str | Path | None = None, *, device: str = Device.DEFAULT) -> Observation:
    data = mutag(path, device=device)
    node_counts = [0] * len(data.node_types)
    bond_counts = [0] * len(data.bond_types)
    nodes, edges, reciprocal = 0, 0, 0
    for graph, node_label, edge_label, _ in (data[index] for index in range(len(data))):
        nodes += graph.nodes
        edges += graph.edges
        for label in node_label.tolist():
            node_counts[label] += 1
        for label in edge_label.tolist():
            bond_counts[label] += 1
        pairs = set(zip(graph.source, graph.target))
        reciprocal += sum((target, source) in pairs for source, target in pairs)
    return Observation(
        device=str(data.node_labels[0].device),
        graphs=len(data),
        classes=2,
        class_counts=tuple(data.labels.count(label) for label in range(2)),
        nodes=nodes,
        minimum_nodes=min(graph.nodes for graph in data.graphs),
        maximum_nodes=max(graph.nodes for graph in data.graphs),
        directed_edges=edges,
        bonds=edges // 2,
        reciprocal_edges=reciprocal,
        node_types=data.node_types,
        node_type_counts=tuple(node_counts),
        bond_types=data.bond_types,
        bond_type_counts=tuple(bond_counts),
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m experiments.mutag_data [path]")
    print(json.dumps(asdict(observe(path)), indent=2))


if __name__ == "__main__":
    main()
