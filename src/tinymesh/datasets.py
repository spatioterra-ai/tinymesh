import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from tinygrad import Tensor
from tinygrad.helpers import fetch

from tinymesh.graph import Graph
from tinymesh.temporal import StaticGraphTemporalSignal

_CHICKENPOX_URL = (
    "https://raw.githubusercontent.com/benedekrozemberczki/"
    "pytorch_geometric_temporal/fe555bc30ee197755c4b58a89407033a5f383415/"
    "dataset/chickenpox.json"
)
_CHICKENPOX_SHA256 = "724b48cfb274b2ecbb855bdb99b970b5ef9dd3671694fa477435dc1e08293735"
_MONTEVIDEO_URL = (
    "https://raw.githubusercontent.com/benedekrozemberczki/"
    "pytorch_geometric_temporal/fe555bc30ee197755c4b58a89407033a5f383415/"
    "dataset/montevideo_bus.json"
)
_MONTEVIDEO_SHA256 = "37d9c6286d474077b5c05173c1570c4da42c387013116daa8862c7a6cab86a75"
_MONTEVIDEO_MAX_BYTES = 4 * 1024 * 1024
_MONTEVIDEO_TIMEOUT = 10


@dataclass(frozen=True)
class _MontevideoSource:
    node_ids: tuple[int, ...]
    source: tuple[int, ...]
    target: tuple[int, ...]
    position: tuple[tuple[float, float], ...]
    road_distance: tuple[float, ...]
    features: tuple[tuple[float, ...], ...]
    targets: tuple[tuple[float, ...], ...]


def chickenpox(
    path: str | Path | None = None,
    *,
    lags: int = 4,
    device: str | None = None,
) -> StaticGraphTemporalSignal:
    """Load the PyG Temporal Hungary chickenpox signal."""
    if not isinstance(lags, int) or isinstance(lags, bool) or lags <= 0:
        raise ValueError("lags must be a positive integer")
    source = Path(path) if path is not None else fetch(_CHICKENPOX_URL, sha256=_CHICKENPOX_SHA256)
    data = json.loads(source.read_bytes())
    edges, node_ids, values = _parse_chickenpox(data)
    if lags >= len(values):
        raise ValueError(f"lags must be smaller than the {len(values)} time steps")

    x = Tensor(
        [
            [[values[time + lag][node] for lag in range(lags)] for node in range(len(node_ids))]
            for time in range(len(values) - lags)
        ],
        device=device,
    ).realize()
    y = Tensor(
        [
            [[values[time + lags][node]] for node in range(len(node_ids))]
            for time in range(len(values) - lags)
        ],
        device=device,
    ).realize()
    graph = Graph(
        len(node_ids),
        [source for source, _ in edges],
        [target for _, target in edges],
    )
    edge_weight = Tensor.ones(graph.edges, dtype=x.dtype, device=x.device).realize()
    return StaticGraphTemporalSignal(graph, node_ids, x, y, edge_weight)


def _read_montevideo(path: str | Path | None = None) -> _MontevideoSource:
    if path is None:
        request = Request(_MONTEVIDEO_URL, headers={"User-Agent": "tinymesh"})
        with urlopen(request, timeout=_MONTEVIDEO_TIMEOUT) as stream:
            payload = stream.read(_MONTEVIDEO_MAX_BYTES + 1)
    else:
        with Path(path).open("rb") as stream:
            payload = stream.read(_MONTEVIDEO_MAX_BYTES + 1)
    if len(payload) > _MONTEVIDEO_MAX_BYTES:
        raise ValueError(f"Montevideo source exceeds {_MONTEVIDEO_MAX_BYTES} bytes")
    if path is None and hashlib.sha256(payload).hexdigest() != _MONTEVIDEO_SHA256:
        raise RuntimeError("Montevideo source checksum mismatch")
    return _parse_montevideo(json.loads(payload))


def _parse_montevideo(data: object) -> _MontevideoSource:
    if not isinstance(data, dict):
        raise TypeError("Montevideo source must be a JSON object")
    if data.get("directed") is not True:
        raise ValueError("Montevideo graph must be directed")
    if data.get("multigraph") is not False:
        raise ValueError("Montevideo graph must not be a multigraph")
    raw_nodes, raw_links = data.get("nodes"), data.get("links")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("nodes must be a non-empty list")
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("links must be a non-empty list")

    node_ids: list[int] = []
    position: list[tuple[float, float]] = []
    features: list[tuple[float, ...]] = []
    targets: list[tuple[float, ...]] = []
    steps: int | None = None
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        node_id = node.get("bus_stop")
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError(f"nodes[{index}].bus_stop must be an integer")
        raw_features = node.get("X")
        if not isinstance(raw_features, dict):
            raise ValueError(f"nodes[{index}].X must be an object")
        feature_values = _series(raw_features.get("y"), f"nodes[{index}].X.y")
        target_values = _series(node.get("y"), f"nodes[{index}].y")
        if len(feature_values) != len(target_values):
            raise ValueError(f"nodes[{index}] feature and target lengths differ")
        if steps is None:
            steps = len(feature_values)
        elif len(feature_values) != steps:
            raise ValueError(f"nodes[{index}] has {len(feature_values)} observations, expected {steps}")
        node_ids.append(node_id)
        position.append(
            (_number(node.get("lon"), f"nodes[{index}].lon"), _number(node.get("lat"), f"nodes[{index}].lat"))
        )
        features.append(feature_values)
        targets.append(target_values)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("bus_stop IDs must be unique")

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    source_rows: list[int] = []
    target_rows: list[int] = []
    road_distance: list[float] = []
    seen: set[tuple[int, int]] = set()
    for index, link in enumerate(raw_links):
        if not isinstance(link, dict):
            raise ValueError(f"links[{index}] must be an object")
        source_id, target_id = link.get("source"), link.get("target")
        if not isinstance(source_id, int) or isinstance(source_id, bool):
            raise ValueError(f"links[{index}] endpoints must be integers")
        if not isinstance(target_id, int) or isinstance(target_id, bool):
            raise ValueError(f"links[{index}] endpoints must be integers")
        if source_id not in node_index or target_id not in node_index:
            raise ValueError(f"links[{index}] endpoint does not resolve")
        edge = node_index[source_id], node_index[target_id]
        if edge in seen:
            raise ValueError(f"links[{index}] duplicates an earlier edge")
        seen.add(edge)
        source_rows.append(edge[0])
        target_rows.append(edge[1])
        road_distance.append(_number(link.get("weight"), f"links[{index}].weight", positive=True))

    return _MontevideoSource(
        tuple(node_ids),
        tuple(source_rows),
        tuple(target_rows),
        tuple(position),
        tuple(road_distance),
        tuple(features),
        tuple(targets),
    )


def _series(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _parse_chickenpox(data: object) -> tuple[list[tuple[int, int]], tuple[str, ...], list[list[float]]]:
    if not isinstance(data, dict):
        raise TypeError("chickenpox source must be a JSON object")
    try:
        raw_edges, raw_node_ids, raw_values = data["edges"], data["node_ids"], data["FX"]
    except KeyError as error:
        raise ValueError(f"chickenpox source is missing {error.args[0]}") from error

    if not isinstance(raw_node_ids, dict) or not all(
        isinstance(name, str) and isinstance(index, int) and not isinstance(index, bool)
        for name, index in raw_node_ids.items()
    ):
        raise ValueError("node_ids must map names to integer rows")
    nodes = len(raw_node_ids)
    if set(raw_node_ids.values()) != set(range(nodes)):
        raise ValueError("node_ids must define contiguous rows from zero")
    node_ids = tuple(name for name, _ in sorted(raw_node_ids.items(), key=lambda item: item[1]))

    if not isinstance(raw_edges, list) or not all(
        isinstance(edge, list)
        and len(edge) == 2
        and all(isinstance(node, int) and not isinstance(node, bool) for node in edge)
        for edge in raw_edges
    ):
        raise ValueError("edges must contain integer source-target pairs")
    edges = [(edge[0], edge[1]) for edge in raw_edges]

    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("FX must contain time-ordered node values")
    if not all(isinstance(row, list) and len(row) == nodes for row in raw_values):
        raise ValueError(f"each FX row must have node width {nodes}")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for row in raw_values
        for value in row
    ):
        raise ValueError("FX values must be numeric")
    values = [[float(value) for value in row] for row in raw_values]
    return edges, node_ids, values
