"""Validate the frozen MBTA topology experiment and its sparse controls."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from tinygrad import Tensor, nn


TOPOLOGY_SEED = 20_260_822
FIXTURE = Path(__file__).parent / "fixtures" / "mbta_topology"


class TopologyError(ValueError):
  """A topology control violates the matched experiment contract."""


@dataclass(frozen=True, order=True)
class Node:
  parent_station: str
  trunk_route_id: str
  direction_id: int


@dataclass(frozen=True, order=True)
class Edge:
  source: Node
  target: Node
  affinity: float


class ResidualModel:
  """A zero-initialized residual MLP shared by every topology arm."""

  def __init__(self, features: int, hidden: int) -> None:
    self.hidden = nn.Linear(features, hidden)
    self.output = nn.Linear(hidden, 1)
    self.output.weight.assign(Tensor.zeros_like(self.output.weight))
    self.output.bias.assign(Tensor.zeros_like(self.output.bias))

  def __call__(self, features: Tensor, anchor: Tensor) -> Tensor:
    return anchor + self.output(self.hidden(features).tanh()).flatten()


@dataclass(frozen=True)
class Observation:
  targets: int
  split: str
  anchor_mae_seconds: float
  self_mae_seconds: float
  true_mae_seconds: float
  reverse_mae_seconds: float
  permuted_mae_seconds: float
  decision: str


def permute_nodes(nodes: tuple[Node, ...], seed: int = TOPOLOGY_SEED) -> dict[Node, Node]:
  """Relabel nodes within route-direction components with one frozen permutation."""
  if len(nodes) != len(set(nodes)):
    raise TopologyError("permutation: duplicate node")
  groups: dict[tuple[str, int], list[Node]] = defaultdict(list)
  for node in nodes:
    groups[node.trunk_route_id, node.direction_id].append(node)
  random = Random(seed)
  result = {}
  for group in sorted(groups):
    source = sorted(groups[group])
    target = source.copy()
    random.shuffle(target)
    result.update(zip(source, target))
  if set(result) != set(nodes) or set(result.values()) != set(nodes):
    raise TopologyError("permutation: node mapping is not bijective")
  return result


def relabel(edges: tuple[Edge, ...], permutation: dict[Node, Node]) -> tuple[Edge, ...]:
  try:
    return tuple(Edge(permutation[edge.source], permutation[edge.target], edge.affinity) for edge in edges)
  except KeyError as error:
    raise TopologyError(f"permutation: unresolved endpoint {error.args[0]!r}") from error


def degree_sequence(edges: tuple[Edge, ...]) -> tuple[tuple[int, int], ...]:
  incoming: dict[Node, int] = defaultdict(int)
  outgoing: dict[Node, int] = defaultdict(int)
  nodes = set()
  for edge in edges:
    if edge.affinity <= 0:
      raise TopologyError("topology: affinity must be positive")
    nodes.update((edge.source, edge.target))
    outgoing[edge.source] += 1
    incoming[edge.target] += 1
  return tuple(sorted((incoming[node], outgoing[node]) for node in nodes))


def observe(path: str | Path = FIXTURE, *, test: bool = False) -> Observation:
  directory = Path(path)
  protocol = _read(directory / "protocol.json")
  validation = _read(directory / "validation.json")
  _validate_protocol(protocol)
  _validate_evidence(validation, "validation", protocol)
  evidence = validation
  if test:
    evidence = _read(directory / "test.json")
    _validate_evidence(evidence, "test", protocol)
    if evidence.get("validation_sha256") != _digest(validation):
      raise TopologyError("test: validation artifact drift")
  learned = {
    arm: sum(result["metrics"]["mae_seconds"] for result in evidence["results"] if result["arm"] == arm) / 3
    for arm in ("self", "true", "reverse", "permuted")
  }
  return Observation(
    targets=evidence["targets"],
    split=evidence["split"],
    anchor_mae_seconds=evidence["baselines"]["anchor"]["mae_seconds"],
    self_mae_seconds=round(learned["self"], 6),
    true_mae_seconds=round(learned["true"], 6),
    reverse_mae_seconds=round(learned["reverse"], 6),
    permuted_mae_seconds=round(learned["permuted"], 6),
    decision=evidence["decision"],
  )


def _validate_protocol(protocol: dict) -> None:
  if protocol.get("schema") != 1 or protocol.get("targets") != 940_551:
    raise TopologyError("protocol: unsupported target population")
  if protocol.get("arms") != ["self", "true", "reverse", "permuted"]:
    raise TopologyError("protocol: topology arms drift")
  if protocol.get("availability") != "retrospective_event_time_only:no_generation_or_ingestion_clock":
    raise TopologyError("protocol: availability drift")


def _validate_evidence(evidence: dict, split: str, protocol: dict) -> None:
  results = evidence.get("results", ())
  identities = {(result.get("arm"), result.get("seed")) for result in results}
  expected = {(arm, seed) for arm in ("self", "true", "reverse", "permuted") for seed in (0, 1, 2)}
  if evidence.get("schema") != 1 or evidence.get("split") != split:
    raise TopologyError(f"{split}: unsupported evidence")
  if evidence.get("protocol_sha256") != _digest(protocol):
    raise TopologyError(f"{split}: protocol artifact drift")
  if identities != expected or any(result.get("metrics", {}).get("targets") != evidence.get("targets") for result in results):
    raise TopologyError(f"{split}: matched arm evidence drift")


def _read(path: Path) -> dict:
  try:
    value = json.loads(path.read_bytes())
  except (FileNotFoundError, json.JSONDecodeError) as error:
    raise TopologyError(f"{path.name}: missing or invalid retained artifact") from error
  if not isinstance(value, dict):
    raise TopologyError(f"{path.name}: retained artifact is not an object")
  return value


def _digest(value: object) -> str:
  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  return hashlib.sha256(encoded.encode()).hexdigest()


def main() -> None:
  print(json.dumps(asdict(observe(test=os.environ.get("TEST") == "1")), indent=2))


if __name__ == "__main__":
  main()
