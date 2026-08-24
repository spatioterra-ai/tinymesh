"""Measure topology assumptions in the public dataset boundaries."""

import json
from collections import deque
from dataclasses import asdict, dataclass
from math import fsum
from statistics import median

from tinygrad import Device

from tinymesh import Graph
from tinymesh.datasets import chickenpox, metr_la, montevideo_bus, mutag


@dataclass(frozen=True)
class Distribution:
  mean: float
  median: float
  p90: int
  maximum: int


@dataclass(frozen=True)
class Topology:
  nodes: int
  edges: int
  unique_edges: int
  duplicate_edges: int
  self_loops: int
  reciprocal_edges: int
  isolated_nodes: int
  source_nodes: int
  sink_nodes: int
  weak_components: int
  largest_weak_component: int
  strong_components: int
  largest_strong_component: int
  in_degree: Distribution
  out_degree: Distribution
  reachable_pairs: int
  possible_pairs: int
  directed_reachability: float | None
  directed_distance: Distribution | None


@dataclass(frozen=True)
class Collection:
  graphs: int
  disconnected_graphs: int
  isolated_nodes: int
  directed_diameter: Distribution


@dataclass(frozen=True)
class Observation:
  chickenpox: Topology
  montevideo: Topology
  metr_la: Topology
  mutag: Collection


def measure(graph: Graph) -> Topology:
  """Return exact, non-self topology measurements without dense adjacency."""
  pairs = set(zip(graph.source, graph.target))
  outgoing = [set() for _ in range(graph.nodes)]
  incoming = [set() for _ in range(graph.nodes)]
  for source, target in pairs:
    if source != target:
      outgoing[source].add(target)
      incoming[target].add(source)

  forward = tuple(tuple(sorted(neighbors)) for neighbors in outgoing)
  reverse = tuple(tuple(sorted(neighbors)) for neighbors in incoming)
  weak = tuple(tuple(sorted(outgoing[node] | incoming[node])) for node in range(graph.nodes))
  weak_sizes = _component_sizes(weak)
  strong_sizes = _strong_component_sizes(forward, reverse)
  distances = _distances(forward)
  possible_pairs = graph.nodes * (graph.nodes - 1)
  in_degree = [len(neighbors) for neighbors in incoming]
  out_degree = [len(neighbors) for neighbors in outgoing]
  return Topology(
    nodes=graph.nodes,
    edges=graph.edges,
    unique_edges=len(pairs),
    duplicate_edges=graph.edges - len(pairs),
    self_loops=sum(source == target for source, target in zip(graph.source, graph.target)),
    reciprocal_edges=sum(source != target and (target, source) in pairs for source, target in pairs),
    isolated_nodes=sum(not weak[node] for node in range(graph.nodes)),
    source_nodes=sum(bool(outgoing[node]) and not incoming[node] for node in range(graph.nodes)),
    sink_nodes=sum(bool(incoming[node]) and not outgoing[node] for node in range(graph.nodes)),
    weak_components=len(weak_sizes),
    largest_weak_component=weak_sizes[0],
    strong_components=len(strong_sizes),
    largest_strong_component=strong_sizes[0],
    in_degree=_distribution(in_degree),
    out_degree=_distribution(out_degree),
    reachable_pairs=len(distances),
    possible_pairs=possible_pairs,
    directed_reachability=len(distances) / possible_pairs if possible_pairs else None,
    directed_distance=_distribution(distances) if distances else None,
  )


def observe(device: str = Device.DEFAULT) -> Observation:
  chickenpox_data = chickenpox(device=device)
  montevideo_data = montevideo_bus(device=device)
  metr_la_data = metr_la(device=device)
  mutag_data = mutag(device=device)
  molecule_topologies = tuple(measure(graph) for graph in mutag_data.graphs)
  molecule_diameters = [
    topology.directed_distance.maximum
    for topology in molecule_topologies
    if topology.directed_distance is not None
  ]
  return Observation(
    chickenpox=measure(chickenpox_data.graph),
    montevideo=measure(montevideo_data.signal.graph),
    metr_la=measure(metr_la_data.graph),
    mutag=Collection(
      graphs=len(molecule_topologies),
      disconnected_graphs=sum(topology.weak_components != 1 for topology in molecule_topologies),
      isolated_nodes=sum(topology.isolated_nodes for topology in molecule_topologies),
      directed_diameter=_distribution(molecule_diameters),
    ),
  )


def _distribution(values: list[int]) -> Distribution:
  ordered = sorted(values)
  return Distribution(
    mean=fsum(ordered) / len(ordered),
    median=median(ordered),
    p90=ordered[(9 * len(ordered) - 1) // 10],
    maximum=ordered[-1],
  )


def _component_sizes(adjacency: tuple[tuple[int, ...], ...]) -> list[int]:
  unseen = set(range(len(adjacency)))
  sizes = []
  while unseen:
    stack = [unseen.pop()]
    size = 0
    while stack:
      node = stack.pop()
      size += 1
      neighbors = unseen.intersection(adjacency[node])
      unseen.difference_update(neighbors)
      stack.extend(neighbors)
    sizes.append(size)
  return sorted(sizes, reverse=True)


def _strong_component_sizes(
  forward: tuple[tuple[int, ...], ...],
  reverse: tuple[tuple[int, ...], ...],
) -> list[int]:
  seen: set[int] = set()
  order = []
  for start in range(len(forward)):
    if start in seen:
      continue
    seen.add(start)
    stack = [(start, 0)]
    while stack:
      node, edge = stack[-1]
      if edge == len(forward[node]):
        order.append(node)
        stack.pop()
        continue
      neighbor = forward[node][edge]
      stack[-1] = node, edge + 1
      if neighbor not in seen:
        seen.add(neighbor)
        stack.append((neighbor, 0))

  unseen = set(range(len(forward)))
  sizes = []
  for start in reversed(order):
    if start not in unseen:
      continue
    unseen.remove(start)
    stack = [start]
    size = 0
    while stack:
      node = stack.pop()
      size += 1
      neighbors = unseen.intersection(reverse[node])
      unseen.difference_update(neighbors)
      stack.extend(neighbors)
    sizes.append(size)
  return sorted(sizes, reverse=True)


def _distances(adjacency: tuple[tuple[int, ...], ...]) -> list[int]:
  distances = []
  for start in range(len(adjacency)):
    reached = {start: 0}
    queue = deque([start])
    while queue:
      node = queue.popleft()
      for neighbor in adjacency[node]:
        if neighbor not in reached:
          reached[neighbor] = reached[node] + 1
          queue.append(neighbor)
    distances.extend(distance for node, distance in reached.items() if node != start)
  return distances


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


if __name__ == "__main__":
  main()
