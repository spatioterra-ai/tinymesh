"""Deterministic topology and trajectories for controlled transport studies."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from random import Random

from tinygrad import Tensor

from tinymesh import Graph


DATA_SEED = 20260729
NODES = 24
TRAIN_TRAJECTORIES = 128
VALIDATION_TRAJECTORIES = 32
TEST_TRAJECTORIES = 32
LOCAL, FORWARD, REVERSE = 0.25, 0.55, 0.20


@dataclass(frozen=True)
class Topology:
  nodes: int
  source: tuple[int, ...]
  target: tuple[int, ...]
  affinity: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class Trajectories:
  values: Tensor

  @property
  def count(self) -> int:
    return int(self.values.shape[0])

  @property
  def steps(self) -> int:
    return int(self.values.shape[1])

  def windows(self, history: int) -> tuple[Tensor, Tensor]:
    if history <= 0 or history >= self.steps:
      raise ValueError(f"history must be in [1, {self.steps})")
    starts = self.steps - history
    values = Tensor.stack(
      *(self.values[:, start : start + history] for start in range(starts)),
      dim=1,
    )
    target = Tensor.stack(
      *(self.values[:, start + history] for start in range(starts)),
      dim=1,
    )
    return (
      values.reshape(self.count * starts, history, *self.values.shape[2:]).realize(),
      target.reshape(self.count * starts, *self.values.shape[2:]).realize(),
    )


def topology(nodes: int) -> Topology:
  if nodes < 8:
    raise ValueError("nodes must be at least eight")
  edges = [(node, (node + 1) % nodes, 1.0 + 0.05 * (node % 5)) for node in range(nodes)]
  edges.extend(
    (
      node,
      (node + 3 + node % 3) % nodes,
      0.25 + 0.05 * (node % 4),
    )
    for node in range(0, nodes, 2)
  )
  return Topology(
    nodes,
    tuple(source for source, _, _ in edges),
    tuple(target for _, target, _ in edges),
    tuple(affinity for _, _, affinity in edges),
  )


def permuted(value: Topology) -> Topology:
  stride = next(candidate for candidate in range(value.nodes - 1, 1, -1) if gcd(candidate, value.nodes) == 1)
  mapping = tuple((stride * node + 1) % value.nodes for node in range(value.nodes))
  return Topology(
    value.nodes,
    tuple(mapping[source] for source in value.source),
    tuple(mapping[target] for target in value.target),
    value.affinity,
  )


def self_topology(value: Topology) -> Topology:
  nodes = tuple(range(value.nodes))
  return Topology(value.nodes, nodes, nodes, (1.0,) * value.nodes)


def symmetric(value: Topology) -> Graph:
  edges = dict.fromkeys(edge for source, target in zip(value.source, value.target) for edge in ((source, target), (target, source)))
  return Graph(
    value.nodes,
    [source for source, _ in edges],
    [target for _, target in edges],
  )


def trajectories(
  value: Topology,
  count: int,
  steps: int,
  seed: int,
  device: str,
  *,
  initial: str = "dense",
) -> Trajectories:
  if initial not in ("dense", "pulse"):
    raise ValueError("initial must be 'dense' or 'pulse'")
  random = Random(seed)
  rows = []
  for _ in range(count):
    values = _initial(value.nodes, random, pulse=initial == "pulse")
    mean = sum(values) / value.nodes
    values = [field - mean for field in values]
    trajectory = [[[field] for field in values]]
    for _ in range(steps - 1):
      values = step(values, value)
      trajectory.append([[field] for field in values])
    rows.append(trajectory)
  return Trajectories(Tensor(rows, device=device).realize())


def _initial(nodes: int, random: Random, *, pulse: bool) -> list[float]:
  if not pulse:
    return [random.uniform(-1, 1) for _ in range(nodes)]
  values = [0.0] * nodes
  selected = random.sample(range(nodes), 4)
  for first, second in zip(selected[::2], selected[1::2]):
    amplitude = random.uniform(0.5, 1)
    values[first], values[second] = amplitude, -amplitude
  return values


def step(values: list[float], value: Topology) -> list[float]:
  outgoing, incoming = [0.0] * value.nodes, [0.0] * value.nodes
  for source, target, affinity in zip(value.source, value.target, value.affinity):
    outgoing[source] += affinity
    incoming[target] += affinity

  forward, reverse = [0.0] * value.nodes, [0.0] * value.nodes
  for source, target, affinity in zip(value.source, value.target, value.affinity):
    forward[target] += affinity / outgoing[source] * values[source]
    reverse[source] += affinity / incoming[target] * values[target]
  return [LOCAL * field + FORWARD * downstream + REVERSE * upstream for field, downstream, upstream in zip(values, forward, reverse)]
