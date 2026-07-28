"""Measure Tinymesh's destination-CSR forward and transpose backward."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from statistics import median

from tinygrad import Device, Tensor
from tinygrad.engine.realize import time_call
from tinygrad.uop.ops import Ops

from tinymesh import Graph


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
    graph = Graph(nodes, source, target)
    values = Tensor.ones(nodes, width, device=device).contiguous().realize()
    gradient = Tensor.ones(nodes, width, device=device).contiguous().realize()
    output = graph.sum(values)
    values_gradient = output.gradient(values, gradient=gradient)[0]

    edges = graph.edges
    empty_targets = nodes - len(set(target))
    empty_sources = nodes - len(set(source))
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
