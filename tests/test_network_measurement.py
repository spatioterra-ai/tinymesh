import unittest

from experiments.network_measurement import measure
from tinymesh import Graph


class NetworkMeasurementTest(unittest.TestCase):
  def test_measures_direction_components_and_exact_distance(self) -> None:
    topology = measure(Graph(4, [0, 0, 1, 1], [1, 1, 1, 2]))

    self.assertEqual((topology.edges, topology.unique_edges, topology.duplicate_edges), (4, 3, 1))
    self.assertEqual((topology.self_loops, topology.reciprocal_edges), (1, 0))
    self.assertEqual((topology.isolated_nodes, topology.source_nodes, topology.sink_nodes), (1, 1, 1))
    self.assertEqual((topology.weak_components, topology.largest_weak_component), (2, 3))
    self.assertEqual((topology.strong_components, topology.largest_strong_component), (4, 1))
    self.assertEqual((topology.reachable_pairs, topology.possible_pairs, topology.directed_reachability), (3, 12, 0.25))
    self.assertEqual(topology.in_degree.maximum, 1)
    self.assertEqual(topology.out_degree.maximum, 1)
    self.assertIsNotNone(topology.directed_distance)
    assert topology.directed_distance is not None
    self.assertAlmostEqual(topology.directed_distance.mean, 4 / 3)
    self.assertEqual((topology.directed_distance.median, topology.directed_distance.p90, topology.directed_distance.maximum), (1, 2, 2))

  def test_self_loops_do_not_hide_structural_isolates(self) -> None:
    topology = measure(Graph(3, [0, 1], [0, 1]))

    self.assertEqual(topology.isolated_nodes, 3)
    self.assertEqual((topology.weak_components, topology.strong_components), (3, 3))
    self.assertEqual((topology.reachable_pairs, topology.possible_pairs), (0, 6))
    self.assertEqual(topology.directed_reachability, 0)
    self.assertIsNone(topology.directed_distance)

  def test_strong_component_counts_mutual_reachability(self) -> None:
    topology = measure(Graph(4, [0, 1, 1, 2, 2], [1, 0, 2, 1, 3]))

    self.assertEqual((topology.weak_components, topology.largest_weak_component), (1, 4))
    self.assertEqual((topology.strong_components, topology.largest_strong_component), (2, 3))
    self.assertEqual(topology.reciprocal_edges, 4)


if __name__ == "__main__":
  unittest.main()
