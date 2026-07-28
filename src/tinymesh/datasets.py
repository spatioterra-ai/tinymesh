import json
from pathlib import Path

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
