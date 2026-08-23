"""Validate event-native memory against the frozen MBTA task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

from tinygrad import Tensor, nn


FIXTURE = Path(__file__).parent / "fixtures" / "mbta_event_memory"
ARMS = ("node_local", "self", "true", "reverse", "permuted")
SEEDS = (0, 1, 2)


class EventMemoryError(ValueError):
  """An event-memory transition or retained artifact violates its boundary."""


@dataclass(frozen=True, order=True)
class Event:
  timestamp: int
  identity: int
  node: int
  value: float


@dataclass(frozen=True)
class Read:
  event: Event
  value: float | None
  elapsed: int | None


@dataclass(frozen=True, order=True)
class Query:
  cutoff: int
  identity: int
  node: int


@dataclass(frozen=True)
class CutoffRead:
  query: Query
  value: float | None
  elapsed: int | None


@dataclass(frozen=True)
class Memory:
  values: tuple[float | None, ...]
  last_update: tuple[int | None, ...]

  @classmethod
  def empty(cls, nodes: int) -> Memory:
    if nodes <= 0:
      raise EventMemoryError("memory: node count must be positive")
    return cls((None,) * nodes, (None,) * nodes)

  def read(self, event: Event) -> Read:
    self._node(event.node)
    updated = self.last_update[event.node]
    if updated is not None and updated >= event.timestamp:
      raise EventMemoryError("memory: read is not strictly after the last update")
    return Read(event, self.values[event.node], None if updated is None else event.timestamp - updated)

  def advance(self, events: tuple[Event, ...]) -> Memory:
    """Apply one equal-time group after every event has read the old state."""
    if not events:
      raise EventMemoryError("memory: update group is empty")
    timestamp = events[0].timestamp
    if any(event.timestamp != timestamp for event in events):
      raise EventMemoryError("memory: update group mixes timestamps")
    grouped: dict[int, list[float]] = {}
    for event in events:
      self._node(event.node)
      updated = self.last_update[event.node]
      if updated is not None and updated >= timestamp:
        raise EventMemoryError("memory: update time did not advance")
      grouped.setdefault(event.node, []).append(event.value)
    values, updates = list(self.values), list(self.last_update)
    for node, observations in grouped.items():
      values[node] = sum(observations) / len(observations)
      updates[node] = timestamp
    return Memory(tuple(values), tuple(updates))

  def _node(self, node: int) -> None:
    if not 0 <= node < len(self.values):
      raise EventMemoryError(f"memory: node {node} is out of range")


def replay(events: tuple[Event, ...], nodes: int) -> tuple[Read, ...]:
  """Read then update timestamp groups from one clean state."""
  ordered = tuple(sorted(events))
  if len({event.identity for event in ordered}) != len(ordered):
    raise EventMemoryError("event: duplicate identity")
  memory = Memory.empty(nodes)
  reads = []
  for _, grouped in groupby(ordered, key=lambda event: event.timestamp):
    group = tuple(grouped)
    reads.extend(memory.read(event) for event in group)
    memory = memory.advance(group)
  return tuple(reads)


def replay_at(events: tuple[Event, ...], queries: tuple[Query, ...], nodes: int) -> tuple[CutoffRead, ...]:
  """Read event state at explicit prediction cutoffs, then continue replay."""
  ordered_events = tuple(sorted(events))
  ordered_queries = tuple(sorted(queries))
  if len({query.identity for query in ordered_queries}) != len(ordered_queries):
    raise EventMemoryError("query: duplicate identity")
  memory = Memory.empty(nodes)
  cursor, reads = 0, []
  for cutoff, grouped in groupby(ordered_queries, key=lambda query: query.cutoff):
    while cursor < len(ordered_events) and ordered_events[cursor].timestamp < cutoff:
      timestamp = ordered_events[cursor].timestamp
      stop = cursor + 1
      while stop < len(ordered_events) and ordered_events[stop].timestamp == timestamp:
        stop += 1
      memory = memory.advance(ordered_events[cursor:stop])
      cursor = stop
    for query in grouped:
      memory._node(query.node)
      updated = memory.last_update[query.node]
      reads.append(CutoffRead(
        query,
        memory.values[query.node],
        None if updated is None else cutoff - updated,
      ))
  return tuple(reads)


class EventEncoder:
  """A small GRU memory updated by an observed event after prediction."""

  def __init__(self, hidden: int, time_features: int) -> None:
    if hidden <= 0 or time_features <= 0:
      raise EventMemoryError("encoder: feature counts must be positive")
    inputs = 1 + time_features
    self.time = nn.Linear(1, time_features)
    self.gates = nn.Linear(inputs + hidden, 2 * hidden)
    self.candidate = nn.Linear(inputs + hidden, hidden)
    self.output = nn.Linear(hidden, 1)
    self.output.weight.assign(Tensor.zeros_like(self.output.weight))
    self.output.bias.assign(Tensor.zeros_like(self.output.bias))
    self.hidden = hidden

  def predict(self, state: Tensor, anchor: Tensor) -> Tensor:
    return anchor + self.output(state).flatten()

  def update(self, elapsed: Tensor, state: Tensor) -> Tensor:
    if elapsed.shape[-1] != 1 or state.shape[-1] != self.hidden:
      raise EventMemoryError("encoder: elapsed or state shape drift")
    encoded = elapsed.cat(self.time(elapsed).cos(), dim=-1)
    gates = self.gates(encoded.cat(state, dim=-1))
    update = gates[..., :self.hidden].sigmoid()
    reset = gates[..., self.hidden:].sigmoid()
    candidate = self.candidate(encoded.cat(state * reset, dim=-1)).tanh()
    return update * state + (1 - update) * candidate


@dataclass(frozen=True)
class Observation:
  targets: int
  split: str
  node_local_mae_seconds: float
  self_mae_seconds: float
  true_mae_seconds: float
  reverse_mae_seconds: float
  permuted_mae_seconds: float
  decision: str


def observe(path: str | Path = FIXTURE) -> Observation:
  directory = Path(path)
  protocol = _read(directory / "protocol.json")
  validation = _read(directory / "validation.json")
  _validate_protocol(protocol)
  _validate_evidence(validation, "validation", protocol)
  means = {
    arm: sum(row["metrics"]["mae_seconds"] for row in validation["results"] if row["arm"] == arm) / len(SEEDS)
    for arm in ARMS
  }
  return Observation(
    targets=validation["targets"],
    split=validation["split"],
    node_local_mae_seconds=round(means["node_local"], 6),
    self_mae_seconds=round(means["self"], 6),
    true_mae_seconds=round(means["true"], 6),
    reverse_mae_seconds=round(means["reverse"], 6),
    permuted_mae_seconds=round(means["permuted"], 6),
    decision=validation["decision"],
  )


def _validate_protocol(protocol: dict) -> None:
  if protocol.get("schema") != 1 or protocol.get("targets") != 940_551:
    raise EventMemoryError("protocol: unsupported target population")
  if protocol.get("arms") != list(ARMS) or protocol.get("seeds") != list(SEEDS):
    raise EventMemoryError("protocol: matched controls drift")
  if protocol.get("prediction_order") != "read_before_event_update":
    raise EventMemoryError("protocol: causal order drift")
  if protocol.get("memory_read_cutoff") != "frozen_source_timestamp_plus_one_second":
    raise EventMemoryError("protocol: memory cutoff drift")
  if protocol.get("state_reset") != "every_service_day_lane_and_independent_replay":
    raise EventMemoryError("protocol: state lifecycle drift")


def _validate_evidence(evidence: dict, split: str, protocol: dict) -> None:
  identities = {(row.get("arm"), row.get("seed")) for row in evidence.get("results", ())}
  expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
  if evidence.get("schema") != 1 or evidence.get("split") != split:
    raise EventMemoryError(f"{split}: unsupported evidence")
  if evidence.get("protocol_sha256") != _digest(protocol):
    raise EventMemoryError(f"{split}: protocol artifact drift")
  if identities != expected or any(row.get("metrics", {}).get("targets") != evidence.get("targets") for row in evidence["results"]):
    raise EventMemoryError(f"{split}: matched arm evidence drift")
  gate = evidence.get("validation_gate", {})
  if split == "validation" and (
    evidence.get("decision") != "stop:event_memory"
    or gate.get("passed") is not False
    or gate.get("selected_arm") != "true"
  ):
    raise EventMemoryError("validation: decision gate drift")


def _read(path: Path) -> dict:
  try:
    value = json.loads(path.read_bytes())
  except (FileNotFoundError, json.JSONDecodeError) as error:
    raise EventMemoryError(f"{path.name}: missing or invalid retained artifact") from error
  if not isinstance(value, dict):
    raise EventMemoryError(f"{path.name}: retained artifact is not an object")
  return value


def _digest(value: object) -> str:
  import hashlib

  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  return hashlib.sha256(encoded.encode()).hexdigest()


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


if __name__ == "__main__":
  main()
