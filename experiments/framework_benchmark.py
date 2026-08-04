"""Compare equal Tinymesh, PyG, and PyG Temporal forward paths."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any, Callable

from tinygrad import Device, Tensor, TinyJit, nn

from tinymesh import Graph
from tinymesh.nn import SAGEConv, TGCN


ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = "2.8.0"
PYG_VERSION = "2.8.0"


@dataclass(frozen=True)
class Timing:
    case: str
    implementation: str
    median_ms: float
    minimum_ms: float
    p90_ms: float


def main() -> None:
    torch, pyg, scatter, PyGSAGE, PyGTTGCN = _torch_stack()
    device = os.getenv("DEV", Device.DEFAULT)
    torch_device = _torch_device(torch, device)
    nodes = _positive("NODES", 4096)
    degree = _positive("DEGREE", 8)
    width = _positive("WIDTH", 32)
    hidden = _positive("HIDDEN", 32)
    warmups = _positive("WARMUPS", 5)
    samples = _positive("SAMPLES", 20)
    if degree >= nodes:
        raise ValueError("DEGREE must be smaller than NODES")

    source, target = _edges(nodes, degree)
    recurrent_source, recurrent_target = _edges(nodes, degree, self_loops=True)
    graph = Graph(nodes, source, target)
    recurrent_graph = Graph(nodes, recurrent_source, recurrent_target)
    values = _values(nodes, width)
    hidden_values = _values(nodes, hidden, offset=7)

    tiny_values = Tensor(values, device=device).reshape(nodes, width).realize()
    tiny_hidden = Tensor(hidden_values, device=device).reshape(nodes, hidden).realize()
    torch_values = torch.tensor(values, dtype=torch.float32, device=torch_device).reshape(nodes, width)
    torch_hidden = torch.tensor(hidden_values, dtype=torch.float32, device=torch_device).reshape(nodes, hidden)
    edge_index = torch.tensor((source, target), dtype=torch.long, device=torch_device)
    recurrent_edge_index = torch.tensor((recurrent_source, recurrent_target), dtype=torch.long, device=torch_device)

    tiny_sage, pyg_sage = SAGEConv(width, hidden), PyGSAGE(width, hidden)
    _configure_sage(tiny_sage, pyg_sage, device, torch)
    tiny_tgcn = TGCN(width, hidden)
    pygt_tgcn = PyGTTGCN(width, hidden, cached=True, add_self_loops=False).to(torch_device)
    _configure_tgcn(tiny_tgcn, pygt_tgcn, device, torch)

    for parameter in nn.state.get_parameters((tiny_sage, tiny_tgcn)):
        parameter.realize()
    pyg_sage = pyg_sage.to(torch_device).eval()
    pygt_tgcn = pygt_tgcn.eval()

    def tiny_sum(value: Tensor) -> Tensor:
        return graph.sum(value).realize()

    def tiny_sage_call(value: Tensor) -> Tensor:
        return tiny_sage(value, graph).realize()

    def tiny_tgcn_call(value: Tensor, state: Tensor) -> Tensor:
        return tiny_tgcn(value, recurrent_graph, state).realize()

    tiny_sum_jit = TinyJit(tiny_sum)
    tiny_sage_jit = TinyJit(tiny_sage_call)
    tiny_tgcn_jit = TinyJit(tiny_tgcn_call)

    @torch.inference_mode()
    def pyg_sum() -> Any:
        return scatter(
            torch_values.index_select(0, edge_index[0]),
            edge_index[1],
            dim=0,
            dim_size=nodes,
            reduce="sum",
        )

    @torch.inference_mode()
    def pyg_sage_call() -> Any:
        return pyg_sage(torch_values, edge_index)

    @torch.inference_mode()
    def pygt_tgcn_call() -> Any:
        return pygt_tgcn(torch_values, recurrent_edge_index, H=torch_hidden)

    parity = {
        "aggregation_max_abs_error": _max_abs(tiny_sum(tiny_values), pyg_sum()),
        "sage_max_abs_error": _max_abs(tiny_sage_call(tiny_values), pyg_sage_call()),
        "tgcn_max_abs_error": _max_abs(tiny_tgcn_call(tiny_values, tiny_hidden), pygt_tgcn_call()),
    }
    if max(parity.values()) > 2e-4:
        raise RuntimeError(f"framework parity failed: {parity}")

    tiny_sync = Device[device].synchronize
    torch_sync = torch.mps.synchronize if torch_device == "mps" else lambda: None
    timings = [
        _time("aggregation", "tinymesh_eager", lambda: tiny_sum(tiny_values), tiny_sync, warmups, samples),
        _time("aggregation", "tinymesh_jit", lambda: tiny_sum_jit(tiny_values), tiny_sync, warmups, samples),
        _time("aggregation", "pyg_eager", pyg_sum, torch_sync, warmups, samples),
        _time("sage", "tinymesh_eager", lambda: tiny_sage_call(tiny_values), tiny_sync, warmups, samples),
        _time("sage", "tinymesh_jit", lambda: tiny_sage_jit(tiny_values), tiny_sync, warmups, samples),
        _time("sage", "pyg_eager", pyg_sage_call, torch_sync, warmups, samples),
        _time("tgcn", "tinymesh_eager", lambda: tiny_tgcn_call(tiny_values, tiny_hidden), tiny_sync, warmups, samples),
        _time("tgcn", "tinymesh_jit", lambda: tiny_tgcn_jit(tiny_values, tiny_hidden), tiny_sync, warmups, samples),
        _time("tgcn", "pygt_eager", pygt_tgcn_call, torch_sync, warmups, samples),
    ]

    print(json.dumps({
        "hardware": {
            "chip": _sysctl("machdep.cpu.brand_string"),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "frameworks": {
            "tinygrad_device": device,
            "torch": torch.__version__,
            "torch_device": torch_device,
            "torch_threads": torch.get_num_threads(),
            "pyg": pyg.__version__,
            "pygt_source": "submodules/pytorch-geometric-temporal",
        },
        "shape": {
            "nodes": nodes,
            "edges": len(source),
            "recurrent_edges": len(recurrent_source),
            "degree": degree,
            "width": width,
            "hidden": hidden,
        },
        "protocol": {
            "mode": "steady-state forward",
            "topology_setup": "excluded",
            "compilation": "excluded by warmup",
            "warmups": warmups,
            "samples": samples,
            "statistic": "median; minimum and p90 retained",
        },
        "parity": parity,
        "parameters": {
            "sage": {
                "tinymesh": _tiny_parameters(tiny_sage),
                "pyg": sum(parameter.numel() for parameter in pyg_sage.parameters()),
            },
            "tgcn": {
                "tinymesh": _tiny_parameters(tiny_tgcn),
                "pygt": sum(parameter.numel() for parameter in pygt_tgcn.parameters()),
                "tinymesh_graph_propagations": 1,
                "pygt_graph_propagations": 3,
            },
        },
        "timings": [asdict(timing) for timing in timings],
    }, indent=2))


def _torch_stack() -> tuple[Any, Any, Callable[..., Any], type, type]:
    try:
        import torch
        import torch_geometric as pyg
        from torch_geometric.nn import SAGEConv as PyGSAGE
        from torch_geometric.utils import scatter
    except ImportError as error:
        raise RuntimeError(
            "run with --with torch==2.8.0 --with torch-geometric==2.8.0"
        ) from error
    if torch.__version__.split("+")[0] != TORCH_VERSION or pyg.__version__ != PYG_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION} and PyG {PYG_VERSION}")

    path = ROOT / "submodules/pytorch-geometric-temporal/torch_geometric_temporal/nn/recurrent/temporalgcn.py"
    if not path.is_file():
        raise RuntimeError("initialize submodules/pytorch-geometric-temporal")
    spec = importlib.util.spec_from_file_location("_tinymesh_pygt_tgcn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, pyg, scatter, PyGSAGE, module.TGCN


def _torch_device(torch: Any, device: str) -> str:
    if device == "CPU":
        return "cpu"
    if device == "METAL" and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("DEV must be CPU, or METAL with PyTorch MPS available")


def _edges(nodes: int, degree: int, *, self_loops: bool = False) -> tuple[list[int], list[int]]:
    source = [node for node in range(nodes) for _ in range(degree)]
    target = [(node + offset) % nodes for node in range(nodes) for offset in range(1, degree + 1)]
    if self_loops:
        source.extend(range(nodes))
        target.extend(range(nodes))
    return source, target


def _values(rows: int, width: int, *, offset: int = 0) -> list[float]:
    return [((index + offset) % 31 - 15) / 31 for index in range(rows * width)]


def _configure_sage(tiny: SAGEConv, pyg: Any, device: str, torch: Any) -> None:
    tiny.neighbor.weight = Tensor.full(tiny.neighbor.weight.shape, 0.01, device=device).realize()
    assert tiny.neighbor.bias is not None
    tiny.neighbor.bias = Tensor.full(tiny.neighbor.bias.shape, 0.03, device=device).realize()
    tiny.root.weight = Tensor.full(tiny.root.weight.shape, 0.02, device=device).realize()
    with torch.no_grad():
        pyg.lin_l.weight.fill_(0.01)
        pyg.lin_l.bias.fill_(0.03)
        pyg.lin_r.weight.fill_(0.02)


def _configure_tgcn(tiny: TGCN, pygt: Any, device: str, torch: Any) -> None:
    width, hidden = tiny.graph_projection.linear.weight.shape[1], tiny.hidden_features
    graph_weight = [
        [scale] * width
        for scale in (0.01, 0.02, 0.03)
        for _ in range(hidden)
    ]
    tiny.graph_projection.linear.weight = Tensor(graph_weight, device=device).realize()
    for scale, gate in zip((0.01, 0.02, 0.03), (tiny.update, tiny.reset, tiny.candidate)):
        gate.weight = Tensor.full(gate.weight.shape, scale, device=device).realize()
        assert gate.bias is not None
        gate.bias = Tensor.full(gate.bias.shape, scale / 10, device=device).realize()

    with torch.no_grad():
        for scale, convolution, gate in zip(
            (0.01, 0.02, 0.03),
            (pygt.conv_z, pygt.conv_r, pygt.conv_h),
            (pygt.linear_z, pygt.linear_r, pygt.linear_h),
        ):
            convolution.lin.weight.fill_(scale)
            convolution.bias.zero_()
            gate.weight.fill_(scale)
            gate.bias.fill_(scale / 10)


def _max_abs(tiny: Tensor, torch: Any) -> float:
    return float(abs(tiny.numpy() - torch.detach().cpu().numpy()).max(initial=0))


def _tiny_parameters(model: object) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _time(
    case: str,
    implementation: str,
    call: Callable[[], object],
    synchronize: Callable[[], None],
    warmups: int,
    samples: int,
) -> Timing:
    for _ in range(warmups):
        call()
        synchronize()
    elapsed = []
    for _ in range(samples):
        synchronize()
        start = perf_counter_ns()
        call()
        synchronize()
        elapsed.append((perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(elapsed)
    return Timing(case, implementation, median(elapsed), ordered[0], ordered[round(0.9 * (samples - 1))])


def _positive(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _sysctl(name: str) -> str | None:
    process = subprocess.run(
        ["sysctl", "-n", name],
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() or None


if __name__ == "__main__":
    main()
