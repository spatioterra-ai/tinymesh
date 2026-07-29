"""Inspect the pinned PyG Temporal Montevideo source."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tinymesh.datasets import _read_montevideo


@dataclass(frozen=True)
class Observation:
    nodes: int
    edges: int
    steps: int
    duplicate_edges: int
    self_loops: int
    position_dimensions: int
    minimum_road_distance: float
    maximum_road_distance: float


def observe(path: str | Path | None = None) -> Observation:
    source = _read_montevideo(path)
    edges = tuple(zip(source.source, source.target))
    return Observation(
        nodes=len(source.node_ids),
        edges=len(edges),
        steps=len(source.features[0]),
        duplicate_edges=len(edges) - len(set(edges)),
        self_loops=sum(source == target for source, target in edges),
        position_dimensions=len(source.position[0]),
        minimum_road_distance=min(source.road_distance),
        maximum_road_distance=max(source.road_distance),
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m experiments.montevideo_source [path]")
    print(json.dumps(asdict(observe(path)), indent=2))


if __name__ == "__main__":
    main()
