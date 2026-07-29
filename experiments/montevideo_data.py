"""Inspect the aligned PyG Temporal Montevideo tensors."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Device

from tinymesh.datasets import montevideo_bus


@dataclass(frozen=True)
class Observation:
    device: str
    coordinate_frame: str
    length_unit: str
    nodes: int
    edges: int
    source_steps: int
    lags: int
    snapshots: int
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    position_shape: tuple[int, ...]
    road_distance_shape: tuple[int, ...]
    first_coordinate_distance: float
    first_road_distance: float


def observe(path: str | Path | None = None, *, lags: int = 4, device: str = Device.DEFAULT) -> Observation:
    data = montevideo_bus(path, lags=lags, device=device)
    signal = data.signal
    x, y = signal[0]
    source = signal.graph.edge_values(data.position, endpoint="source")
    target = signal.graph.edge_values(data.position, endpoint="target")
    coordinate_distance = ((target - source) ** 2).sum(axis=-1).sqrt().realize()
    return Observation(
        device=signal.x.device,
        coordinate_frame=data.coordinate_frame,
        length_unit=data.length_unit,
        nodes=signal.graph.nodes,
        edges=signal.graph.edges,
        source_steps=len(signal) + lags,
        lags=lags,
        snapshots=len(signal),
        x_shape=x.shape,
        y_shape=y.shape,
        position_shape=data.position.shape,
        road_distance_shape=data.road_distance.shape,
        first_coordinate_distance=float(coordinate_distance[0].item()),
        first_road_distance=float(data.road_distance[0].item()),
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m experiments.montevideo_data [path]")
    print(json.dumps(asdict(observe(path)), indent=2))


if __name__ == "__main__":
    main()
