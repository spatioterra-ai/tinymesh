"""Measure Tinygrad's native edge-list aggregation against Tinymesh's sparse gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from random import Random

from tinygrad import Device, GlobalCounters, Tensor


@dataclass(frozen=True)
class Observation:
    nodes: int
    edges: int
    width: int
    operations: int
    kernels: int
    dense_carrier_count: int


def native_edge_sum(
    node_state: Tensor,
    source: Tensor,
    target: Tensor,
    node_count: int,
) -> Tensor:
    """Gather source rows, then sum edge messages into target rows."""
    if node_state.ndim != 2:
        raise ValueError(f"node_state must have shape [N, H], got {node_state.shape}")
    if source.ndim != 1 or source.shape != target.shape:
        raise ValueError("source and target must have the same shape [E]")

    source_index = source.reshape(-1, 1).expand(source.shape[0], node_state.shape[1])
    messages = node_state.gather(0, source_index)
    index = target.reshape(-1, 1).expand(messages.shape)
    return Tensor.zeros(
        node_count,
        node_state.shape[1],
        dtype=node_state.dtype,
        device=node_state.device,
    ).scatter_reduce(0, index, messages, "sum", include_self=False)


def measure(
    nodes: int,
    edges: int,
    width: int,
    *,
    device: str,
    seed: int = 7,
) -> Observation:
    rng = Random(seed)
    state = Tensor(
        [
            [float((node * width + feature) % 17) for feature in range(width)]
            for node in range(nodes)
        ],
        device=device,
    ).realize()
    source = Tensor([rng.randrange(nodes) for _ in range(edges)], device=device).realize()
    target = Tensor([rng.randrange(nodes) for _ in range(edges)], device=device).realize()
    output = native_edge_sum(state, source, target, nodes)
    dense_carrier_count = _dense_carrier_count(output, nodes, edges, width)

    GlobalCounters.reset()
    output.realize()
    return Observation(
        nodes,
        edges,
        width,
        GlobalCounters.global_ops,
        GlobalCounters.kernel_count,
        dense_carrier_count,
    )


def _dense_carrier_count(
    output: Tensor,
    nodes: int,
    edges: int,
    width: int,
) -> int:
    # UOps are inspected only by this revision-bound experiment, never by the package.
    dimensions = sorted((nodes, edges, width))
    return sum(
        1
        for uop in output.uop.toposort()
        if len(shape := tuple(int(size) for size in uop.shape)) == 3
        and sorted(shape) == dimensions
    )


def _tinygrad_revision() -> str:
    direct_url = distribution("tinygrad").read_text("direct_url.json")
    if direct_url is None:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


def main() -> None:
    observations = [
        measure(nodes, 2 * nodes, 4, device=Device.DEFAULT)
        for nodes in (16, 32, 64, 128)
    ]
    growth = [
        current.operations / previous.operations
        for previous, current in zip(observations, observations[1:])
    ]
    blocked = (
        any(observation.dense_carrier_count for observation in observations)
        or growth[-1] > 3.0
    )
    print(json.dumps({
        "tinygrad_revision": _tinygrad_revision(),
        "device": Device.DEFAULT,
        "candidate": "advanced indexing + scatter_reduce(sum)",
        "observations": [asdict(observation) for observation in observations],
        "operation_growth_when_N_and_E_double": growth,
        "representative_logical_N_E_H_lanes": 30_000 * 60_000 * 32,
        "decision": "blocked" if blocked else "revisit",
    }, indent=2))


if __name__ == "__main__":
    main()
