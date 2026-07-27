"""Probe edge-linear CSR aggregation and its transpose backward in Tinygrad."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from statistics import median

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import time_call
from tinygrad.uop.ops import KernelInfo, Ops


@dataclass(frozen=True, init=False)
class CSRTopology:
    nodes: int
    row_ptr: tuple[int, ...]
    column: tuple[int, ...]
    transpose_row_ptr: tuple[int, ...]
    transpose_column: tuple[int, ...]

    def __init__(
        self,
        nodes: int,
        source: list[int] | tuple[int, ...],
        target: list[int] | tuple[int, ...],
    ) -> None:
        if nodes <= 0:
            raise ValueError("nodes must be positive")
        if len(source) != len(target):
            raise ValueError("source and target must have the same length")
        if any(node < 0 or node >= nodes for edge in (source, target) for node in edge):
            raise ValueError(f"node IDs must be in [0, {nodes})")

        column, row_ptr = _group(nodes, target, source)
        transpose_column, transpose_row_ptr = _group(nodes, source, target)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "row_ptr", row_ptr)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "transpose_row_ptr", transpose_row_ptr)
        object.__setattr__(self, "transpose_column", transpose_column)

    def _tensors(self, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return tuple(
            Tensor(values, dtype=dtypes.int32, device=device).realize()
            for values in (
                self.row_ptr,
                self.column,
                self.transpose_row_ptr,
                self.transpose_column,
            )
        )


@dataclass(frozen=True)
class Observation:
    topology: str
    nodes: int
    edges: int
    width: int
    topology_elements: int
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
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = [[] for _ in range(nodes)]
    for row, column in zip(owner, neighbor):
        rows[row].append(column)
    flat = tuple(column for row in rows for column in sorted(row))
    row_ptr = [0]
    for row in rows:
        row_ptr.append(row_ptr[-1] + len(row))
    return flat, tuple(row_ptr)


def _csr_sum_kernel(output: UOp, values: UOp, row_ptr: UOp, column: UOp, *_: UOp) -> UOp:
    nodes, width = output.shape
    output, values, row_ptr, column = output.base, values.base, row_ptr.base, column.base
    lane = UOp.range(nodes * width, 0, dtype=dtypes.int32)
    row, feature = lane // width, lane % width
    start, stop = row_ptr[row].cast(dtypes.int32), row_ptr[row + 1].cast(dtypes.int32)

    accumulator = UOp.placeholder((1,), values.dtype, 0, addrspace=AddrSpace.REG)
    edge = UOp.placeholder((1,), dtypes.int32, 1, addrspace=AddrSpace.REG)
    accumulator_init = accumulator.after(lane)[0].store(0.0)
    edge_init = edge.after(accumulator_init)[0].store(start)
    loop = UOp(Ops.LOOP, src=(edge_init,), arg=(1,))

    current = edge.after(loop)[0].load()
    active = current < stop
    source = column[active.where(current, 0)].cast(dtypes.int32)
    message = values[source * width + feature]
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
