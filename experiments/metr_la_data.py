"""Inspect the aligned METR-LA sensor tensors."""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Device

from tinymesh.datasets import metr_la


@dataclass(frozen=True)
class Observation:
    device: str
    nodes: int
    edges: int
    steps: int
    first_timestamp: str
    last_timestamp: str
    sample_minutes: int
    values: int
    observed_values: int
    missing_values: int
    self_loops: int
    asymmetric_edges: int
    minimum_affinity: float
    maximum_affinity: float
    affinity_sha256: str


def observe(path: str | Path | None = None, *, device: str = Device.DEFAULT) -> Observation:
    data = metr_la(path, device=device)
    pairs = set(zip(data.graph.source, data.graph.target))
    values = data.speed.numel()
    observed = int(data.observed.sum().item())
    return Observation(
        device=data.speed.device,
        nodes=data.graph.nodes,
        edges=data.graph.edges,
        steps=data.speed.shape[0],
        first_timestamp=data.timestamps[0].isoformat(" "),
        last_timestamp=data.timestamps[-1].isoformat(" "),
        sample_minutes=data.sample_minutes,
        values=values,
        observed_values=observed,
        missing_values=values - observed,
        self_loops=sum(source == target for source, target in pairs),
        asymmetric_edges=sum((target, source) not in pairs for source, target in pairs),
        minimum_affinity=float(data.affinity.min().item()),
        maximum_affinity=float(data.affinity.max().item()),
        affinity_sha256=hashlib.sha256(data.affinity.data().tobytes()).hexdigest(),
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m experiments.metr_la_data [path]")
    print(json.dumps(asdict(observe(path)), indent=2))


if __name__ == "__main__":
    main()
