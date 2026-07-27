"""Probe edge-linear CSR aggregation and its transpose backward in Tinygrad."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from importlib.metadata import distribution
from statistics import median

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import time_call
from tinygrad.uop.ops import AxisType, KernelInfo, Ops


@dataclass(frozen=True, init=False)
class CSRTopology:
    nodes: int
    source: tuple[int, ...]
    target: tuple[int, ...]
    row_ptr: tuple[int, ...]
    column: tuple[int, ...]
    transpose_row_ptr: tuple[int, ...]
    transpose_column: tuple[int, ...]
    edge_order: tuple[int, ...]
    transpose_order: tuple[int, ...]
    _tensors_by_device: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _degree_by_device: dict[str, Tensor] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _edge_tensors_by_device: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        nodes: int,
        source: list[int] | tuple[int, ...],
        target: list[int] | tuple[int, ...],
    ) -> None:
        source, target = tuple(source), tuple(target)
        if nodes <= 0:
            raise ValueError("nodes must be positive")
        if len(source) != len(target):
            raise ValueError("source and target must have the same length")
        if any(node < 0 or node >= nodes for edge in (source, target) for node in edge):
            raise ValueError(f"node IDs must be in [0, {nodes})")

        column, row_ptr, edge_order = _group(nodes, target, source)
        transpose_column, transpose_row_ptr, transpose_order = _group(nodes, source, target)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "row_ptr", row_ptr)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "transpose_row_ptr", transpose_row_ptr)
        object.__setattr__(self, "transpose_column", transpose_column)
        object.__setattr__(self, "edge_order", edge_order)
        object.__setattr__(self, "transpose_order", transpose_order)
        object.__setattr__(self, "_tensors_by_device", {})
        object.__setattr__(self, "_degree_by_device", {})
        object.__setattr__(self, "_edge_tensors_by_device", {})

    def _tensors(self, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = self._tensors_by_device.get(device)
        if tensors is None:
            tensors = (
                Tensor(self.row_ptr, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.column, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_row_ptr, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_column, dtype=dtypes.int32, device=device).realize(),
            )
            self._tensors_by_device[device] = tensors
        return tensors

    def _degree(self, device: str) -> Tensor:
        degree = self._degree_by_device.get(device)
        if degree is None:
            degree = Tensor(
                [stop - start for start, stop in zip(self.row_ptr, self.row_ptr[1:])],
                dtype=dtypes.int32,
                device=device,
            ).realize()
            self._degree_by_device[device] = degree
        return degree

    def _edge_tensors(self, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = self._edge_tensors_by_device.get(device)
        if tensors is None:
            tensors = (
                Tensor(self.edge_order, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_order, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.source, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.target, dtype=dtypes.int32, device=device).realize(),
            )
            self._edge_tensors_by_device[device] = tensors
        return tensors


@dataclass(frozen=True)
class Observation:
    topology: str
    nodes: int
    edges: int
    width: int
    csr_elements: int
    forward_lane_work: int
    backward_lane_work: int
    forward_median_ms: float
    backward_median_ms: float


def main() -> None:
    sizes = tuple(int(value) for value in os.getenv("SIZES", "4096,8192,16384").split(","))
    degree = int(os.getenv("DEGREE", "8"))
    width = int(os.getenv("WIDTH", "32"))
    warmups = int(os.getenv("WARMUPS", "5"))
    samples = int(os.getenv("SAMPLES", "20"))
    observations = [
        measure("balanced", nodes, degree, width, Device.DEFAULT, warmups, samples)
        for nodes in sizes
    ]
    observations.extend(
        measure(kind, sizes[-1], degree, width, Device.DEFAULT, warmups, samples)
        for kind in ("destination_hub", "source_hub")
    )
    print(json.dumps({
        "tinygrad_revision": _tinygrad_revision(),
        "device": Device.DEFAULT,
        "arch": getattr(Device[Device.DEFAULT], "arch", None),
        "candidate": "destination CSR forward + transpose CSR backward",
        "kernel_optimization": "disabled (opts_to_apply=())",
        "observations": [asdict(observation) for observation in observations],
    }, indent=2))


def measure(
    kind: str,
    nodes: int,
    degree: int,
    width: int,
    device: str,
    warmups: int = 5,
    samples: int = 20,
) -> Observation:
    source, target = _edges(kind, nodes, degree)
    topology = CSRTopology(nodes, source, target)
    values = Tensor.ones(nodes, width, device=device).contiguous().realize()
    gradient = Tensor.ones(nodes, width, device=device).contiguous().realize()
    output = csr_edge_sum(values, topology)
    values_gradient = output.gradient(values, gradient=gradient)[0]

    empty_targets = _empty_rows(topology.row_ptr)
    empty_sources = _empty_rows(topology.transpose_row_ptr)
    edges = len(source)
    return Observation(
        kind,
        nodes,
        edges,
        width,
        2 * edges + 2 * (nodes + 1),
        (nodes + edges + empty_targets) * width,
        (nodes + edges + empty_sources) * width,
        _median_ms(output, warmups, samples),
        _median_ms(values_gradient, warmups, samples),
    )


def _group(
    nodes: int,
    owner: list[int] | tuple[int, ...],
    neighbor: list[int] | tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows: list[list[tuple[int, int]]] = [[] for _ in range(nodes)]
    for edge, (row_index, column) in enumerate(zip(owner, neighbor)):
        rows[row_index].append((column, edge))
    flat = tuple(item for row in rows for item in sorted(row))
    row_ptr = [0]
    for row in rows:
        row_ptr.append(row_ptr[-1] + len(row))
    return (
        tuple(column for column, _ in flat),
        tuple(row_ptr),
        tuple(edge for _, edge in flat),
    )


def _csr_sum_kernel(output: UOp, values: UOp, row_ptr: UOp, column: UOp, *_: UOp) -> UOp:
    return _csr_kernel(output, values, row_ptr, column)


def _csr_reindexed_weighted_sum_kernel(
    output: UOp,
    values: UOp,
    edge_weight: UOp,
    row_ptr: UOp,
    column: UOp,
    weight_index: UOp,
    *_: UOp,
) -> UOp:
    return _csr_kernel(output, values, row_ptr, column, edge_weight, weight_index)


def _csr_kernel(
    output: UOp,
    values: UOp,
    row_ptr: UOp,
    column: UOp,
    edge_weight: UOp | None = None,
    weight_index: UOp | None = None,
) -> UOp:
    nodes, width = output.shape
    output, values, row_ptr, column = output.base, values.base, row_ptr.base, column.base
    if edge_weight is not None:
        edge_weight = edge_weight.base
    if weight_index is not None:
        weight_index = weight_index.base
    lane = UOp.range(nodes * width, 0, dtype=dtypes.int32)
    row, feature = lane // width, lane % width
    start, stop = row_ptr[row].cast(dtypes.int32), row_ptr[row + 1].cast(dtypes.int32)

    accumulator = UOp.placeholder((1,), values.dtype, 0, addrspace=AddrSpace.REG)
    edge = UOp.placeholder((1,), dtypes.int32, 1, addrspace=AddrSpace.REG)
    accumulator_init = accumulator.after(lane)[0].store(0.0)
    accumulator = accumulator.after(accumulator_init)
    edge_init = edge.after(accumulator_init)[0].store(start)
    edge = edge.after(edge_init)
    loop = UOp.loop(1)

    current = edge.after(loop)[0].load()
    active = current < stop
    position = active.where(current, 0)
    source = column[position].load().cast(dtypes.int32)
    message = values[source * width + feature]
    if edge_weight is not None:
        edge_index = position if weight_index is None else weight_index[position].load().cast(dtypes.int32)
        message = message * edge_weight[edge_index]
    next_edge = current + 1
    updated = UOp.group(
        accumulator[0].store(
            accumulator.after(loop)[0].load()
            + active.where(message, 0.0)
        ),
        edge[0].store(next_edge),
    )
    done = updated.end(loop, next_edge < stop)
    return output[lane].store(accumulator.after(done)[0].load()).end(lane).sink(
        arg=KernelInfo(name="csr_sum", opts_to_apply=())
    )


def _edge_dot_kernel(
    output: UOp,
    values: UOp,
    gradient: UOp,
    source: UOp,
    target: UOp,
) -> UOp:
    edges, width = output.shape[0], values.shape[1]
    edge = UOp.range(edges, 0, dtype=dtypes.int32)
    feature = UOp.range(width, 1, axis_type=AxisType.REDUCE)
    edge_source = source[edge].load().cast(dtypes.int32)
    edge_target = target[edge].load().cast(dtypes.int32)
    output = output[edge].set(0.0)
    output = output[edge].set(
        output.after(feature)[edge] + values[edge_source, feature] * gradient[edge_target, feature],
        end=feature,
    )
    return output.end(edge).sink(arg=KernelInfo(name="edge_dot", opts_to_apply=()))


def _csr_sum_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    _, values, _, _, transpose_row_ptr, transpose_column = call.src[1:]
    output = Tensor.invalids(*values.shape, dtype=values.dtype, device=values.device)
    grad_values = output.custom_kernel(
        Tensor(gradient),
        Tensor(transpose_row_ptr),
        Tensor(transpose_column),
        fxn=_csr_sum_kernel,
    )[0]
    return None, grad_values.uop, None, None, None, None


def _csr_weighted_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    (
        _,
        values,
        edge_weight,
        _,
        _,
        _,
        transpose_row_ptr,
        transpose_column,
        transpose_order,
        source,
        target,
    ) = call.src[1:]
    values_output = Tensor.invalids(*values.shape, dtype=values.dtype, device=values.device)
    grad_values = values_output.custom_kernel(
        Tensor(gradient),
        Tensor(edge_weight),
        Tensor(transpose_row_ptr),
        Tensor(transpose_column),
        Tensor(transpose_order),
        fxn=_csr_reindexed_weighted_sum_kernel,
    )[0]
    weight_output = Tensor.invalids(*edge_weight.shape, dtype=edge_weight.dtype, device=edge_weight.device)
    grad_weight = weight_output.custom_kernel(
        Tensor(values),
        Tensor(gradient),
        Tensor(source),
        Tensor(target),
        fxn=_edge_dot_kernel,
    )[0]
    return None, grad_values.uop, grad_weight.uop, None, None, None, None, None, None, None, None


def csr_edge_sum(
    values: Tensor,
    topology: CSRTopology,
) -> Tensor:
    if values.ndim != 2:
        raise ValueError(f"values must have shape [N, H], got {values.shape}")
    if values.shape[0] != topology.nodes:
        raise ValueError(f"values must have {topology.nodes} rows, got {values.shape[0]}")
    if not isinstance(values.device, str):
        raise ValueError("CSR aggregation requires one device")
    if topology.nodes == 1:
        return values * len(topology.column)
    if not topology.column:
        return values * 0
    row_ptr, column, transpose_row_ptr, transpose_column = topology._tensors(values.device)
    output = Tensor.invalids(values.shape[0], values.shape[1], dtype=values.dtype, device=values.device)
    return output.custom_kernel(
        values,
        row_ptr,
        column,
        transpose_row_ptr,
        transpose_column,
        fxn=_csr_sum_kernel,
        grad_fxn=_csr_sum_gradient,
    )[0]


def csr_edge_weighted_sum(
    values: Tensor,
    topology: CSRTopology,
    edge_weight: Tensor,
) -> Tensor:
    if values.ndim != 2:
        raise ValueError(f"values must have shape [N, H], got {values.shape}")
    if values.shape[0] != topology.nodes:
        raise ValueError(f"values must have {topology.nodes} rows, got {values.shape[0]}")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != len(topology.column):
        raise ValueError(f"edge_weight must have shape [{len(topology.column)}], got {edge_weight.shape}")
    if values.dtype != edge_weight.dtype:
        raise ValueError(f"values and edge_weight must have the same dtype, got {values.dtype} and {edge_weight.dtype}")
    if not isinstance(values.device, str) or edge_weight.device != values.device:
        raise ValueError("weighted CSR aggregation requires one shared device")
    if topology.nodes == 1 or not topology.column:
        return values * edge_weight.sum()
    row_ptr, column, transpose_row_ptr, transpose_column = topology._tensors(values.device)
    edge_order, transpose_order, source, target = topology._edge_tensors(values.device)
    output = Tensor.invalids(values.shape[0], values.shape[1], dtype=values.dtype, device=values.device)
    return output.custom_kernel(
        values,
        edge_weight,
        row_ptr,
        column,
        edge_order,
        transpose_row_ptr,
        transpose_column,
        transpose_order,
        source,
        target,
        fxn=_csr_reindexed_weighted_sum_kernel,
        grad_fxn=_csr_weighted_gradient,
    )[0]


def _edges(kind: str, nodes: int, degree: int) -> tuple[list[int], list[int]]:
    if kind == "balanced":
        return (
            [node for node in range(nodes) for _ in range(degree)],
            [(node + offset + 1) % nodes for node in range(nodes) for offset in range(degree)],
        )
    if kind == "destination_hub":
        return [node for node in range(nodes) for _ in range(degree)], [0] * (nodes * degree)
    if kind == "source_hub":
        return [0] * (nodes * degree), [node for node in range(nodes) for _ in range(degree)]
    raise ValueError(f"unknown topology {kind!r}")


def _empty_rows(row_ptr: tuple[int, ...]) -> int:
    return sum(start == stop for start, stop in zip(row_ptr, row_ptr[1:]))


def _median_ms(tensor: Tensor, warmups: int, samples: int) -> float:
    linear = tensor.schedule_linear()
    calls = [call for call in linear.src if call.src[0].op is Ops.SINK]
    if len(calls) != 1:
        raise RuntimeError(f"expected one CSR kernel, got {len(calls)}")
    for _ in range(warmups):
        time_call(calls[0])
    return median(time_call(calls[0]) for _ in range(samples)) * 1_000


def _tinygrad_revision() -> str:
    direct_url = distribution("tinygrad").read_text("direct_url.json")
    if direct_url is None:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


if __name__ == "__main__":
    main()
