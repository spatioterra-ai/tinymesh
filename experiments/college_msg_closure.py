"""Measure temporal triadic closure in the CollegeMsg interaction stream."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable


SOURCE_BYTES = 345_339
SOURCE_SHA256 = "50ae2d98ed3bad9ddb18dbd495a89e5e10cfb8f7e86932827db29fc41b41f9fa"


@dataclass(frozen=True, order=True)
class Interaction:
  timestamp: int
  source: int
  target: int


@dataclass(frozen=True)
class SourceAudit:
  messages: int
  nodes: int
  directed_pairs: int
  undirected_pairs: int
  self_messages: int
  first_timestamp: int
  last_timestamp: int


@dataclass(frozen=True)
class ClosureAudit:
  first_contacts: int
  entry_contacts: int
  repeat_messages: int
  wedge_formations: int
  non_wedge_formations: int
  wedge_pair_seconds: int
  non_wedge_pair_seconds: int
  wedge_rate_per_million_pair_days: float | None
  non_wedge_rate_per_million_pair_days: float | None
  rate_ratio: float | None


@dataclass(frozen=True)
class Observation:
  source: SourceAudit
  closure: ClosureAudit
  relation: str
  projection: str
  decision: str


def load(path: str | Path) -> tuple[Interaction, ...]:
  """Load the exact checksum-pinned CollegeMsg gzip source."""
  source = Path(path)
  payload = source.read_bytes()
  if len(payload) != SOURCE_BYTES or hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("source does not match the pinned CollegeMsg artifact")
  with gzip.open(source, "rt", encoding="ascii") as lines:
    return parse(lines)


def parse(lines: Iterable[str]) -> tuple[Interaction, ...]:
  """Parse sorted ``source target timestamp`` rows without changing identity."""
  interactions = []
  previous = -1
  for line_number, line in enumerate(lines, 1):
    fields = line.split()
    if len(fields) != 3:
      raise ValueError(f"line {line_number}: expected source target timestamp")
    try:
      source, target, timestamp = map(int, fields)
    except ValueError as error:
      raise ValueError(f"line {line_number}: fields must be integers") from error
    if source < 0 or target < 0 or timestamp < 0:
      raise ValueError(f"line {line_number}: identifiers and timestamp must be non-negative")
    if timestamp < previous:
      raise ValueError(f"line {line_number}: timestamp moved backward")
    interactions.append(Interaction(timestamp, source, target))
    previous = timestamp
  if not interactions:
    raise ValueError("source contains no interactions")
  return tuple(interactions)


def observe(interactions: tuple[Interaction, ...]) -> Observation:
  """Measure first-contact incidence from graph state strictly before each time."""
  nodes = {node for interaction in interactions for node in (interaction.source, interaction.target)}
  directed = {(interaction.source, interaction.target) for interaction in interactions if interaction.source != interaction.target}
  source = SourceAudit(
    messages=len(interactions),
    nodes=len(nodes),
    directed_pairs=len(directed),
    undirected_pairs=len({_pair(source, target) for source, target in directed}),
    self_messages=sum(interaction.source == interaction.target for interaction in interactions),
    first_timestamp=interactions[0].timestamp,
    last_timestamp=interactions[-1].timestamp,
  )
  closure = _measure_closure(interactions)
  return Observation(
    source=source,
    closure=closure,
    relation="directed_message",
    projection="undirected_first_contact",
    decision="retain:research_only",
  )


def _measure_closure(interactions: tuple[Interaction, ...]) -> ClosureAudit:
  known: set[int] = set()
  edges: set[tuple[int, int]] = set()
  neighbors: dict[int, set[int]] = {}
  open_wedges: set[tuple[int, int]] = set()
  entry_contacts = wedge_formations = non_wedge_formations = 0
  wedge_pair_seconds = non_wedge_pair_seconds = 0
  previous = interactions[0].timestamp

  for timestamp, group in groupby(interactions, key=lambda interaction: interaction.timestamp):
    elapsed = timestamp - previous
    possible = len(known) * (len(known) - 1) // 2 - len(edges)
    wedge_pairs = len(open_wedges)
    wedge_pair_seconds += wedge_pairs * elapsed
    non_wedge_pair_seconds += (possible - wedge_pairs) * elapsed

    messages = tuple(group)
    new_pairs = {
      _pair(interaction.source, interaction.target)
      for interaction in messages
      if interaction.source != interaction.target and _pair(interaction.source, interaction.target) not in edges
    }
    for pair in new_pairs:
      if pair[0] not in known or pair[1] not in known:
        entry_contacts += 1
      elif pair in open_wedges:
        wedge_formations += 1
      else:
        non_wedge_formations += 1

    for source, target in sorted(new_pairs):
      _add_edge(source, target, edges, neighbors, open_wedges)
    known.update(node for interaction in messages for node in (interaction.source, interaction.target))
    previous = timestamp

  first_contacts = len(edges)
  repeat_messages = len(interactions) - sum(interaction.source == interaction.target for interaction in interactions) - first_contacts
  wedge_rate = _rate(wedge_formations, wedge_pair_seconds)
  non_wedge_rate = _rate(non_wedge_formations, non_wedge_pair_seconds)
  ratio = wedge_rate / non_wedge_rate if wedge_rate is not None and non_wedge_rate not in (None, 0) else None
  return ClosureAudit(
    first_contacts=first_contacts,
    entry_contacts=entry_contacts,
    repeat_messages=repeat_messages,
    wedge_formations=wedge_formations,
    non_wedge_formations=non_wedge_formations,
    wedge_pair_seconds=wedge_pair_seconds,
    non_wedge_pair_seconds=non_wedge_pair_seconds,
    wedge_rate_per_million_pair_days=wedge_rate,
    non_wedge_rate_per_million_pair_days=non_wedge_rate,
    rate_ratio=ratio,
  )


def _add_edge(
  source: int,
  target: int,
  edges: set[tuple[int, int]],
  neighbors: dict[int, set[int]],
  open_wedges: set[tuple[int, int]],
) -> None:
  source_neighbors = neighbors.setdefault(source, set())
  target_neighbors = neighbors.setdefault(target, set())
  for neighbor in source_neighbors:
    candidate = _pair(target, neighbor)
    if candidate not in edges:
      open_wedges.add(candidate)
  for neighbor in target_neighbors:
    candidate = _pair(source, neighbor)
    if candidate not in edges:
      open_wedges.add(candidate)
  edge = _pair(source, target)
  edges.add(edge)
  open_wedges.discard(edge)
  source_neighbors.add(target)
  target_neighbors.add(source)


def _pair(source: int, target: int) -> tuple[int, int]:
  return (source, target) if source < target else (target, source)


def _rate(formations: int, pair_seconds: int) -> float | None:
  if pair_seconds == 0:
    return None
  return formations * 86_400_000_000 / pair_seconds


def main() -> None:
  path = os.environ.get("SOURCE")
  if path is None:
    raise SystemExit("SOURCE must name the checksum-pinned CollegeMsg.txt.gz")
  print(json.dumps(asdict(observe(load(path))), indent=2))


if __name__ == "__main__":
  main()
