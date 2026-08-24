"""Measure temporal triadic closure in the CollegeMsg interaction stream."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from itertools import groupby

from tinymesh import TemporalEdges
from tinymesh.datasets import college_msg


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


def observe(interactions: TemporalEdges) -> Observation:
  """Measure first-contact incidence from graph state strictly before each time."""
  if interactions.edges == 0:
    raise ValueError("closure measurement requires at least one interaction")
  directed = {(source, target) for source, target in zip(interactions.source, interactions.target) if source != target}
  source = SourceAudit(
    messages=interactions.edges,
    nodes=interactions.nodes,
    directed_pairs=len(directed),
    undirected_pairs=len({_pair(source, target) for source, target in directed}),
    self_messages=sum(source == target for source, target in zip(interactions.source, interactions.target)),
    first_timestamp=interactions.timestamp[0],
    last_timestamp=interactions.timestamp[-1],
  )
  closure = _measure_closure(interactions)
  return Observation(
    source=source,
    closure=closure,
    relation="directed_message",
    projection="undirected_first_contact",
    decision="retain:closure_research_only",
  )


def _measure_closure(interactions: TemporalEdges) -> ClosureAudit:
  known: set[int] = set()
  edges: set[tuple[int, int]] = set()
  neighbors: dict[int, set[int]] = {}
  open_wedges: set[tuple[int, int]] = set()
  entry_contacts = wedge_formations = non_wedge_formations = 0
  wedge_pair_seconds = non_wedge_pair_seconds = 0
  previous = interactions.timestamp[0]

  events = zip(interactions.timestamp, interactions.source, interactions.target)
  for timestamp, group in groupby(events, key=lambda event: event[0]):
    elapsed = timestamp - previous
    possible = len(known) * (len(known) - 1) // 2 - len(edges)
    wedge_pairs = len(open_wedges)
    wedge_pair_seconds += wedge_pairs * elapsed
    non_wedge_pair_seconds += (possible - wedge_pairs) * elapsed

    messages = tuple(group)
    new_pairs = {
      _pair(source, target)
      for _, source, target in messages
      if source != target and _pair(source, target) not in edges
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
    known.update(node for _, source, target in messages for node in (source, target))
    previous = timestamp

  first_contacts = len(edges)
  repeat_messages = interactions.edges - sum(
    source == target for source, target in zip(interactions.source, interactions.target)
  ) - first_contacts
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
  print(json.dumps(asdict(observe(college_msg(path).events)), indent=2))


if __name__ == "__main__":
  main()
