import csv
import hashlib
import json
import math
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from tinygrad import Tensor, dtypes
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
_METR_LA_TRAFFIC = (
    "METR-LA.csv",
    "https://zenodo.org/api/records/5724362/files/METR-LA.csv/content",
    72_467_662,
    "8d67a35472db1719d7d4be851f2bf64cb21d9c52577c8a6b4b873d43205af381",
)
_DCRNN_REVISION = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
_METR_LA_SENSOR_IDS = (
    "graph_sensor_ids.txt",
    f"https://raw.githubusercontent.com/liyaguang/DCRNN/{_DCRNN_REVISION}/"
    "data/sensor_graph/graph_sensor_ids.txt",
    1_448,
    "3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990",
)
_METR_LA_DISTANCES = (
    "distances_la_2012.csv",
    f"https://raw.githubusercontent.com/liyaguang/DCRNN/{_DCRNN_REVISION}/"
    "data/sensor_graph/distances_la_2012.csv",
    6_393_348,
    "a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5",
)
_METR_LA_MAX_BYTES = {
    "METR-LA.csv": 80 * 1024 * 1024,
    "graph_sensor_ids.txt": 4 * 1024,
    "distances_la_2012.csv": 8 * 1024 * 1024,
}
_METR_LA_STEP = timedelta(minutes=5)
_METR_LA_THRESHOLD = 0.1
_MUTAG_SOURCE = (
    "MUTAG.zip",
    "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip",
    24_550,
    "c419bdc853c367d2d83da4973c45100954ae15e10f5ae2cddde6ca431f8207f6",
)
_MUTAG_MAX_BYTES = 32 * 1024
_MUTAG_MEMBER_MAX_BYTES = {
    "graph_labels": 1024,
    "node_labels": 8 * 1024,
    "graph_indicator": 16 * 1024,
    "A": 96 * 1024,
    "edge_labels": 16 * 1024,
}
_MUTAG_NODE_TYPES = ("C", "N", "O", "F", "I", "Cl", "Br")
_MUTAG_BOND_TYPES = ("aromatic", "single", "double", "triple")


@dataclass(frozen=True, eq=False)
class MontevideoBus:
    """The aligned PyG Temporal Montevideo bus signal."""

    signal: StaticGraphTemporalSignal
    position: Tensor
    road_distance: Tensor
    coordinate_frame: str = field(init=False, default="EPSG:32721")
    length_unit: str = field(init=False, default="m")

    def __post_init__(self) -> None:
        if self.signal.edge_weight is not None:
            raise ValueError("Montevideo signal edge_weight must be None")
        if self.position.shape != (self.signal.graph.nodes, 2):
            raise ValueError(f"position must have shape [{self.signal.graph.nodes}, 2], got {self.position.shape}")
        if self.road_distance.shape != (self.signal.graph.edges,):
            raise ValueError(
                f"road_distance must have shape [{self.signal.graph.edges}], got {self.road_distance.shape}"
            )
        for name, value in (("position", self.position), ("road_distance", self.road_distance)):
            if value.dtype != self.signal.x.dtype or value.device != self.signal.x.device:
                raise ValueError(f"{name} must share signal dtype and device")


@dataclass(frozen=True, eq=False)
class METRLA:
    """Raw METR-LA traffic speed aligned with the directed DCRNN graph."""

    graph: Graph
    sensor_ids: tuple[str, ...]
    timestamps: tuple[datetime, ...]
    speed: Tensor
    affinity: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_ids", tuple(self.sensor_ids))
        object.__setattr__(self, "timestamps", tuple(self.timestamps))
        if self.speed.ndim != 2:
            raise ValueError(f"speed must have shape [T, N], got {self.speed.shape}")
        if self.speed.shape[1] != self.graph.nodes:
            raise ValueError(f"speed must have {self.graph.nodes} node columns, got {self.speed.shape[1]}")
        if len(self.sensor_ids) != self.graph.nodes:
            raise ValueError(f"expected {self.graph.nodes} sensor IDs, got {len(self.sensor_ids)}")
        if any(not sensor_id for sensor_id in self.sensor_ids):
            raise ValueError("sensor IDs must be non-empty")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor IDs must be unique")
        if len(self.timestamps) != self.speed.shape[0]:
            raise ValueError(f"expected {self.speed.shape[0]} timestamps, got {len(self.timestamps)}")
        if any(not isinstance(timestamp, datetime) or timestamp.tzinfo is not None for timestamp in self.timestamps):
            raise ValueError("timestamps must be naive datetimes")
        if any(right - left != _METR_LA_STEP for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be five minutes apart")
        if not dtypes.is_float(self.speed.dtype):
            raise ValueError(f"speed must have a floating dtype, got {self.speed.dtype}")
        if self.affinity.ndim != 1 or self.affinity.shape[0] != self.graph.edges:
            raise ValueError(f"affinity must have shape [{self.graph.edges}], got {self.affinity.shape}")
        if not dtypes.is_float(self.affinity.dtype):
            raise ValueError(f"affinity must have a floating dtype, got {self.affinity.dtype}")
        if self.speed.dtype != self.affinity.dtype or self.speed.device != self.affinity.device:
            raise ValueError("speed and affinity must share dtype and device")

    @property
    def observed(self) -> Tensor:
        """Mask the zero sentinel used by the reference METR-LA protocol."""
        return self.speed != 0

    @property
    def sample_minutes(self) -> int:
        return 5


@dataclass(frozen=True, eq=False)
class MUTAG:
    """MUTAG molecular graphs with categorical atom, bond, and graph labels."""

    graphs: tuple[Graph, ...]
    node_labels: tuple[Tensor, ...]
    edge_labels: tuple[Tensor, ...]
    labels: tuple[int, ...]
    node_types: tuple[str, ...] = field(init=False, default=_MUTAG_NODE_TYPES)
    bond_types: tuple[str, ...] = field(init=False, default=_MUTAG_BOND_TYPES)

    def __post_init__(self) -> None:
        for name in ("graphs", "node_labels", "edge_labels", "labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.graphs or len({len(self.graphs), len(self.node_labels), len(self.edge_labels), len(self.labels)}) != 1:
            raise ValueError("MUTAG fields must contain the same positive number of graphs")
        if any(isinstance(label, bool) or label not in (0, 1) for label in self.labels):
            raise ValueError("MUTAG labels must be 0 or 1")
        for index, (graph, node_label, edge_label) in enumerate(zip(self.graphs, self.node_labels, self.edge_labels)):
            if node_label.ndim != 1 or node_label.shape[0] != graph.nodes:
                raise ValueError(f"node_labels[{index}] must have shape [{graph.nodes}], got {node_label.shape}")
            if edge_label.ndim != 1 or edge_label.shape[0] != graph.edges:
                raise ValueError(f"edge_labels[{index}] must have shape [{graph.edges}], got {edge_label.shape}")
            if not dtypes.is_int(node_label.dtype) or not dtypes.is_int(edge_label.dtype):
                raise ValueError("MUTAG node and edge labels must be integer tensors")
            if node_label.device != edge_label.device or not isinstance(node_label.device, str):
                raise ValueError("each MUTAG graph requires one shared device")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> tuple[Graph, Tensor, Tensor, int]:
        return self.graphs[index], self.node_labels[index], self.edge_labels[index], self.labels[index]


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


def montevideo_bus(
    path: str | Path | None = None,
    *,
    lags: int = 4,
    device: str | None = None,
) -> MontevideoBus:
    """Load the PyG Temporal Montevideo bus signal without normalization."""
    if not isinstance(lags, int) or isinstance(lags, bool) or lags <= 0:
        raise ValueError("lags must be a positive integer")
    source = _read_montevideo(path)
    steps = len(source.features[0])
    if lags >= steps:
        raise ValueError(f"lags must be smaller than the {steps} time steps")

    features = Tensor(source.features, device=device).T
    targets = Tensor(source.targets, dtype=features.dtype, device=features.device).T
    snapshots = steps - lags
    x = Tensor.stack(*(features[lag:lag + snapshots] for lag in range(lags)), dim=2).realize()
    y = targets[lags:].unsqueeze(2).realize()
    graph = Graph(len(source.node_ids), source.source, source.target)
    signal = StaticGraphTemporalSignal(graph, tuple(str(node_id) for node_id in source.node_ids), x, y)
    position = Tensor(source.position, dtype=x.dtype, device=x.device).realize()
    road_distance = Tensor(source.road_distance, dtype=x.dtype, device=x.device).realize()
    return MontevideoBus(signal, position, road_distance)


def mutag(path: str | Path | None = None, *, device: str | None = None) -> MUTAG:
    """Load the TU Dortmund MUTAG molecular graph collection."""
    try:
        with ZipFile(BytesIO(_read_mutag(path))) as archive:
            return _parse_mutag(archive, device)
    except BadZipFile as error:
        raise ValueError("MUTAG source must be a ZIP archive") from error


def metr_la(
    path: str | Path | None = None,
    *,
    device: str | None = None,
) -> METRLA:
    """Load raw METR-LA speed and reproduce the directed DCRNN affinity."""
    traffic_path, sensor_path, distance_path = _metr_la_sources(path)
    sensor_ids = _read_sensor_ids(sensor_path)
    timestamps, values = _read_traffic(traffic_path, sensor_ids)
    graph, affinity_values = _read_road_graph(distance_path, sensor_ids)
    speed = _float_tensor(values, (len(timestamps), len(sensor_ids)), device)
    if not isinstance(speed.device, str):
        raise ValueError("METR-LA requires one device")
    affinity = _float_tensor(affinity_values, (graph.edges,), speed.device)
    return METRLA(graph, sensor_ids, timestamps, speed, affinity)


def _metr_la_sources(path: str | Path | None) -> tuple[Path, Path, Path]:
    if path is not None:
        root = Path(path)
        if not root.is_dir():
            raise ValueError("METR-LA path must be a directory")
        sources = root / _METR_LA_TRAFFIC[0], root / _METR_LA_SENSOR_IDS[0], root / _METR_LA_DISTANCES[0]
        for source in sources:
            if source.stat().st_size > _METR_LA_MAX_BYTES[source.name]:
                raise ValueError(f"{source.name} exceeds {_METR_LA_MAX_BYTES[source.name]} bytes")
        return sources
    return _fetch_source(_METR_LA_TRAFFIC), _fetch_source(_METR_LA_SENSOR_IDS), _fetch_source(_METR_LA_DISTANCES)


def _fetch_source(source: tuple[str, str, int, str]) -> Path:
    name, url, size, sha256 = source
    path = fetch(url, name=f"{sha256[:12]}-{name}", subdir="tinymesh", sha256=sha256)
    if path.stat().st_size != size:
        raise RuntimeError(f"{name} must contain {size} bytes")
    return path


def _read_sensor_ids(path: Path) -> tuple[str, ...]:
    sensor_ids = tuple(path.read_text().strip().split(","))
    if not sensor_ids or any(not sensor_id for sensor_id in sensor_ids):
        raise ValueError("sensor IDs must be non-empty")
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("sensor IDs must be unique")
    return sensor_ids


def _read_traffic(path: Path, sensor_ids: tuple[str, ...]) -> tuple[tuple[datetime, ...], array[float]]:
    timestamps: list[datetime] = []
    values = array("f")
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        try:
            header = next(rows)
        except StopIteration as error:
            raise ValueError("METR-LA traffic CSV must not be empty") from error
        if not header or header[0] or tuple(header[1:]) != sensor_ids:
            raise ValueError("traffic columns must match sensor ID order")
        for index, row in enumerate(rows, start=2):
            if len(row) != len(sensor_ids) + 1:
                raise ValueError(f"traffic row {index} must have {len(sensor_ids) + 1} columns")
            try:
                timestamp = datetime.fromisoformat(row[0])
            except ValueError as error:
                raise ValueError(f"traffic row {index} has an invalid timestamp") from error
            if timestamp.tzinfo is not None:
                raise ValueError(f"traffic row {index} timestamp must be naive")
            if timestamps and timestamp - timestamps[-1] != _METR_LA_STEP:
                raise ValueError(f"traffic row {index} must be five minutes after the previous row")
            timestamps.append(timestamp)
            for column, raw in enumerate(row[1:], start=2):
                try:
                    value = float(raw)
                except ValueError as error:
                    raise ValueError(f"traffic row {index} column {column} must be numeric") from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"traffic row {index} column {column} must be finite and non-negative")
                values.append(value)
    if len(timestamps) < 2:
        raise ValueError("METR-LA traffic CSV must contain at least two rows")
    return tuple(timestamps), values


def _read_road_graph(path: Path, sensor_ids: tuple[str, ...]) -> tuple[Graph, array[float]]:
    node_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
    distances: dict[tuple[int, int], float] = {}
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["from", "to", "cost"]:
            raise ValueError("distance columns must be from,to,cost")
        for index, row in enumerate(rows, start=2):
            if row["from"] not in node_index or row["to"] not in node_index:
                continue
            edge = node_index[row["from"]], node_index[row["to"]]
            if edge in distances:
                raise ValueError(f"distance row {index} duplicates an earlier edge")
            try:
                distance = float(row["cost"])
            except ValueError as error:
                raise ValueError(f"distance row {index} cost must be numeric") from error
            if not math.isfinite(distance) or distance < 0:
                raise ValueError(f"distance row {index} cost must be finite and non-negative")
            distances[edge] = distance
    if any(distances.get((node, node)) != 0 for node in range(len(sensor_ids))):
        raise ValueError("every sensor must have a zero-distance self edge")

    mean = math.fsum(distances.values()) / len(distances)
    scale = math.sqrt(math.fsum((distance - mean) ** 2 for distance in distances.values()) / len(distances))
    if scale == 0:
        raise ValueError("road distances must have non-zero variance")
    edges = []
    for (source, target), distance in sorted(distances.items()):
        affinity = math.exp(-(distance / scale) ** 2)
        if affinity >= _METR_LA_THRESHOLD:
            edges.append((source, target, affinity))
    graph = Graph(len(sensor_ids), [source for source, _, _ in edges], [target for _, target, _ in edges])
    return graph, array("f", (affinity for _, _, affinity in edges))


def _float_tensor(values: array[float], shape: tuple[int, ...], device: str | None) -> Tensor:
    if values.itemsize != 4:
        raise RuntimeError("native float storage must be four bytes")
    return Tensor(values.tobytes(), dtype=dtypes.float32, device=device).reshape(*shape).realize()


def _parse_mutag(archive: ZipFile, device: str | None) -> MUTAG:
    indicator = tuple(value - 1 for value in _mutag_integers(archive, "graph_indicator"))
    node_labels = _mutag_integers(archive, "node_labels")
    graph_labels = _mutag_integers(archive, "graph_labels")
    edge_labels = _mutag_integers(archive, "edge_labels")
    edges = _mutag_edges(archive)

    if not indicator or indicator[0] != 0 or any(left > right for left, right in zip(indicator, indicator[1:])):
        raise ValueError("MUTAG graph indicators must be ordered from one")
    graphs = indicator[-1] + 1
    if set(indicator) != set(range(graphs)):
        raise ValueError("MUTAG graph indicators must be contiguous")
    if len(graph_labels) != graphs:
        raise ValueError(f"MUTAG must contain one label for each of its {graphs} graphs")
    if len(node_labels) != len(indicator):
        raise ValueError("MUTAG must contain one label for each node")
    if len(edge_labels) != len(edges):
        raise ValueError("MUTAG must contain one label for each edge")
    if any(label not in range(len(_MUTAG_NODE_TYPES)) for label in node_labels):
        raise ValueError("MUTAG node labels must be in [0, 7)")
    if any(label not in range(len(_MUTAG_BOND_TYPES)) for label in edge_labels):
        raise ValueError("MUTAG edge labels must be in [0, 4)")
    if any(label not in (-1, 1) for label in graph_labels):
        raise ValueError("MUTAG graph labels must be -1 or 1")

    labeled_edges: dict[tuple[int, int], int] = {}
    edge_groups: list[list[tuple[int, int, int]]] = [[] for _ in range(graphs)]
    starts = [0]
    for index in range(1, len(indicator)):
        if indicator[index] != indicator[index - 1]:
            starts.append(index)
    for index, ((source, target), label) in enumerate(zip(edges, edge_labels), start=1):
        if source < 0 or source >= len(indicator) or target < 0 or target >= len(indicator):
            raise ValueError(f"MUTAG edge {index} endpoint is out of range")
        if source == target:
            raise ValueError(f"MUTAG edge {index} is a self-loop")
        if indicator[source] != indicator[target]:
            raise ValueError(f"MUTAG edge {index} crosses graph boundaries")
        if (source, target) in labeled_edges:
            raise ValueError(f"MUTAG edge {index} duplicates an earlier edge")
        labeled_edges[source, target] = label
        graph = indicator[source]
        edge_groups[graph].append((source - starts[graph], target - starts[graph], label))
    if any(labeled_edges.get((target, source)) != label for (source, target), label in labeled_edges.items()):
        raise ValueError("MUTAG bonds must contain matching edges in both directions")

    ends = [*starts[1:], len(indicator)]
    graph_objects, node_tensors, edge_tensors = [], [], []
    for graph, (start, end) in enumerate(zip(starts, ends)):
        group = edge_groups[graph]
        graph_objects.append(Graph(end - start, [source for source, _, _ in group], [target for _, target, _ in group]))
        node_tensors.append(Tensor(node_labels[start:end], device=device).realize())
        edge_tensors.append(Tensor([label for _, _, label in group], device=device).realize())
    return MUTAG(
        tuple(graph_objects),
        tuple(node_tensors),
        tuple(edge_tensors),
        tuple(int(label == 1) for label in graph_labels),
    )


def _read_mutag(path: str | Path | None) -> bytes:
    if path is None:
        request = Request(_MUTAG_SOURCE[1], headers={"User-Agent": "tinymesh"})
        with urlopen(request, timeout=10) as stream:
            payload = stream.read(_MUTAG_MAX_BYTES + 1)
    else:
        with Path(path).open("rb") as stream:
            payload = stream.read(_MUTAG_MAX_BYTES + 1)
    if len(payload) > _MUTAG_MAX_BYTES:
        raise ValueError(f"MUTAG source exceeds {_MUTAG_MAX_BYTES} bytes")
    if path is None and (len(payload) != _MUTAG_SOURCE[2] or hashlib.sha256(payload).hexdigest() != _MUTAG_SOURCE[3]):
        raise RuntimeError("MUTAG source identity mismatch")
    return payload


def _mutag_integers(archive: ZipFile, name: str) -> tuple[int, ...]:
    values = []
    for index, row in enumerate(_mutag_rows(archive, name), start=1):
        try:
            values.append(int(row))
        except ValueError as error:
            raise ValueError(f"MUTAG {name} row {index} must be an integer") from error
    return tuple(values)


def _mutag_edges(archive: ZipFile) -> tuple[tuple[int, int], ...]:
    edges = []
    for index, row in enumerate(_mutag_rows(archive, "A"), start=1):
        values = row.split(",")
        if len(values) != 2:
            raise ValueError(f"MUTAG A row {index} must contain two endpoints")
        try:
            edges.append((int(values[0].strip()) - 1, int(values[1].strip()) - 1))
        except ValueError as error:
            raise ValueError(f"MUTAG A row {index} endpoints must be integers") from error
    return tuple(edges)


def _mutag_rows(archive: ZipFile, name: str) -> tuple[str, ...]:
    member = f"MUTAG/MUTAG_{name}.txt"
    matches = [info for info in archive.infolist() if info.filename == member]
    if len(matches) != 1:
        raise ValueError(f"MUTAG source must contain {member} exactly once")
    info = matches[0]
    if info.file_size > _MUTAG_MEMBER_MAX_BYTES[name]:
        raise ValueError(f"MUTAG {name} exceeds {_MUTAG_MEMBER_MAX_BYTES[name]} bytes")
    try:
        rows = tuple(row.strip() for row in archive.read(info).decode("ascii").splitlines())
    except UnicodeDecodeError as error:
        raise ValueError(f"MUTAG {name} must be ASCII") from error
    if not rows or any(not row for row in rows):
        raise ValueError(f"MUTAG {name} must contain non-empty rows")
    return rows


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
