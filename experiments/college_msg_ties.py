"""Measure tie strength, neighborhood overlap, and fragmentation in CollegeMsg."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from tinymesh import TemporalEdges
from tinymesh.datasets import college_msg


REMOVAL_PERCENTS = tuple(range(0, 101, 10))


@dataclass(frozen=True)
class Tie:
  pair: tuple[int, int]
  messages: int
  active_days: int
  reciprocal: bool
  embeddedness: int
  overlap: float | None


@dataclass(frozen=True)
class GroupAudit:
  label: str
  edges: int
  overlap_edges: int
  mean_overlap: float | None
  mean_embeddedness: float | None
  local_bridges: int


@dataclass(frozen=True)
class MetricAudit:
  metric: str
  groups: tuple[GroupAudit, ...]


@dataclass(frozen=True)
class FragmentPoint:
  removed_percent: int
  removed_edges: int
  components: int
  largest_component: int
  largest_fraction: float


@dataclass(frozen=True)
class FragmentCurve:
  order: str
  points: tuple[FragmentPoint, ...]
  mean_largest_fraction: float


@dataclass(frozen=True)
class FragmentAudit:
  metric: str
  weak_first: FragmentCurve
  strong_first: FragmentCurve


@dataclass(frozen=True)
class Observation:
  nodes: int
  messages: int
  self_messages: int
  ties: int
  reciprocal_ties: int
  local_bridges: int
  undefined_overlaps: int
  strength: tuple[MetricAudit, ...]
  reciprocity: MetricAudit
  fragmentation: tuple[FragmentAudit, ...]
  random_baseline: FragmentCurve
  seed: int
  relation: str
  projection: str
  decision: str


def observe(interactions: TemporalEdges, seed: int = 0) -> Observation:
  """Measure final observed contacts without promoting a social-tie ontology."""
  if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
    raise ValueError("seed must be a non-negative integer")
  ties, self_messages = _project(interactions)
  if not ties:
    raise ValueError("tie measurement requires at least one non-self interaction")
  random_order = _shuffled(ties, seed)
  return Observation(
    nodes=interactions.nodes,
    messages=interactions.edges,
    self_messages=self_messages,
    ties=len(ties),
    reciprocal_ties=sum(tie.reciprocal for tie in ties),
    local_bridges=sum(tie.embeddedness == 0 for tie in ties),
    undefined_overlaps=sum(tie.overlap is None for tie in ties),
    strength=(
      _metric(ties, "message_count", lambda tie: tie.messages),
      _metric(ties, "active_days", lambda tie: tie.active_days),
    ),
    reciprocity=_reciprocity(ties),
    fragmentation=tuple(
      _fragmentation(interactions.nodes, ties, name, key, seed)
      for name, key in (
        ("message_count", lambda tie: tie.messages),
        ("active_days", lambda tie: tie.active_days),
        ("reciprocity", lambda tie: int(tie.reciprocal)),
      )
    ),
    random_baseline=_curve(interactions.nodes, random_order, "seeded_random"),
    seed=seed,
    relation="directed_message",
    projection="final_undirected_contact",
    decision="retain:tie_structure_research_only",
  )


def _project(interactions: TemporalEdges) -> tuple[tuple[Tie, ...], int]:
  counts: dict[tuple[int, int], int] = {}
  days: dict[tuple[int, int], set[int]] = {}
  directions: dict[tuple[int, int], set[tuple[int, int]]] = {}
  self_messages = 0
  for source, target, timestamp in zip(interactions.source, interactions.target, interactions.timestamp):
    if source == target:
      self_messages += 1
      continue
    pair = _pair(source, target)
    counts[pair] = counts.get(pair, 0) + 1
    days.setdefault(pair, set()).add(timestamp // 86_400)
    directions.setdefault(pair, set()).add((source, target))

  neighbors = [set() for _ in range(interactions.nodes)]
  for source, target in counts:
    neighbors[source].add(target)
    neighbors[target].add(source)

  ties = []
  for pair in sorted(counts):
    source, target = pair
    common = neighbors[source] & neighbors[target]
    external = (neighbors[source] | neighbors[target]) - {source, target}
    ties.append(Tie(
      pair,
      counts[pair],
      len(days[pair]),
      len(directions[pair]) == 2,
      len(common),
      len(common) / len(external) if external else None,
    ))
  return tuple(ties), self_messages


def _metric(ties: tuple[Tie, ...], name: str, value: Callable[[Tie], int]) -> MetricAudit:
  groups: dict[int, list[Tie]] = {}
  for tie in ties:
    lower = 1 << (value(tie).bit_length() - 1)
    groups.setdefault(lower, []).append(tie)
  return MetricAudit(
    name,
    tuple(
      _group(f"{lower}" if lower == 1 else f"{lower}-{2 * lower - 1}", groups[lower])
      for lower in sorted(groups)
    ),
  )


def _reciprocity(ties: tuple[Tie, ...]) -> MetricAudit:
  return MetricAudit(
    "reciprocity",
    tuple(
      _group(label, [tie for tie in ties if tie.reciprocal is reciprocal])
      for label, reciprocal in (("one_way", False), ("reciprocal", True))
    ),
  )


def _group(label: str, ties: list[Tie]) -> GroupAudit:
  overlaps = [tie.overlap for tie in ties if tie.overlap is not None]
  return GroupAudit(
    label,
    len(ties),
    len(overlaps),
    sum(overlaps) / len(overlaps) if overlaps else None,
    sum(tie.embeddedness for tie in ties) / len(ties) if ties else None,
    sum(tie.embeddedness == 0 for tie in ties),
  )


def _fragmentation(
  nodes: int,
  ties: tuple[Tie, ...],
  name: str,
  value: Callable[[Tie], int],
  seed: int,
) -> FragmentAudit:
  shuffled = _shuffled(ties, seed)
  return FragmentAudit(
    name,
    _curve(nodes, tuple(sorted(shuffled, key=value)), "weak_first"),
    _curve(nodes, tuple(sorted(shuffled, key=value, reverse=True)), "strong_first"),
  )


def _curve(nodes: int, removal_order: tuple[Tie, ...], order: str) -> FragmentCurve:
  points = []
  for percent in REMOVAL_PERCENTS:
    removed = len(removal_order) * percent // 100
    components, largest = _components(nodes, removal_order[removed:])
    points.append(FragmentPoint(percent, removed, components, largest, largest / nodes))
  return FragmentCurve(order, tuple(points), sum(point.largest_fraction for point in points) / len(points))


def _components(nodes: int, ties: tuple[Tie, ...]) -> tuple[int, int]:
  neighbors = [[] for _ in range(nodes)]
  for tie in ties:
    source, target = tie.pair
    neighbors[source].append(target)
    neighbors[target].append(source)

  seen: set[int] = set()
  components = largest = 0
  for root in range(nodes):
    if root in seen:
      continue
    components += 1
    seen.add(root)
    stack = [root]
    size = 0
    while stack:
      node = stack.pop()
      size += 1
      for neighbor in neighbors[node]:
        if neighbor not in seen:
          seen.add(neighbor)
          stack.append(neighbor)
    largest = max(largest, size)
  return components, largest


def _shuffled(ties: tuple[Tie, ...], seed: int) -> tuple[Tie, ...]:
  shuffled = list(ties)
  random.Random(seed).shuffle(shuffled)
  return tuple(shuffled)


def _pair(source: int, target: int) -> tuple[int, int]:
  return (source, target) if source < target else (target, source)


def main() -> None:
  path = os.environ.get("SOURCE")
  seed = int(os.environ.get("SEED", "0"))
  print(json.dumps(asdict(observe(college_msg(path).events, seed)), indent=2))


if __name__ == "__main__":
  main()
