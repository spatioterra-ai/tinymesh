import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tinygrad import Tensor

from experiments.mbta_topology import FIXTURE, Edge, Node, ResidualModel, TopologyError, degree_sequence, observe, permute_nodes, relabel


class MbtaTopologyTest(unittest.TestCase):
  def test_permutation_is_component_local_bijective_and_degree_preserving(self) -> None:
    nodes = tuple(Node(f"s{index}", route, direction) for route in ("Blue", "Red") for direction in (0, 1) for index in range(4))
    edges = tuple(
      Edge(nodes[offset + index], nodes[offset + index + 1], index + 1)
      for offset in range(0, len(nodes), 4)
      for index in range(3)
    )

    permutation = permute_nodes(nodes)
    changed = relabel(edges, permutation)

    self.assertEqual(set(permutation), set(nodes))
    self.assertEqual(set(permutation.values()), set(nodes))
    self.assertTrue(all(
      (source.trunk_route_id, source.direction_id) == (target.trunk_route_id, target.direction_id)
      for source, target in permutation.items()
    ))
    self.assertNotEqual(changed, edges)
    self.assertEqual(degree_sequence(changed), degree_sequence(edges))
    self.assertEqual(changed, relabel(edges, permute_nodes(nodes)))

  def test_malformed_topology_controls_fail(self) -> None:
    node = Node("s0", "Red", 0)
    with self.assertRaisesRegex(TopologyError, "duplicate node"):
      permute_nodes((node, node))
    with self.assertRaisesRegex(TopologyError, "unresolved endpoint"):
      relabel((Edge(node, Node("s1", "Red", 0), 1),), {node: node})
    with self.assertRaisesRegex(TopologyError, "affinity must be positive"):
      degree_sequence((Edge(node, node, 0),))

  def test_zero_initialized_residual_starts_at_the_shared_anchor(self) -> None:
    model = ResidualModel(3, 4)
    anchor = Tensor([2.0, 5.0])

    self.assertEqual(model(Tensor.zeros(2, 3), anchor).tolist(), anchor.tolist())

  def test_retained_validation_is_exact_and_learned_test_stays_closed(self) -> None:
    result = observe()

    self.assertEqual((result.targets, result.split), (138_910, "validation"))
    self.assertEqual(result.anchor_mae_seconds, 152.175644)
    self.assertEqual(
      (result.self_mae_seconds, result.true_mae_seconds, result.reverse_mae_seconds, result.permuted_mae_seconds),
      (150.271906, 145.792679, 150.682343, 149.895777),
    )
    self.assertEqual(result.decision, "freeze:open_learned_test_once")
    self.assertFalse((FIXTURE / "test.json").exists())
    self.assertEqual(result, observe())

  def test_retained_artifacts_fail_closed_on_drift(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      protocol_path = fixture / "protocol.json"
      protocol = json.loads(protocol_path.read_text())
      protocol["targets"] += 1
      protocol_path.write_text(json.dumps(protocol))
      with self.assertRaisesRegex(TopologyError, "unsupported target population"):
        observe(fixture)

    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      validation_path = fixture / "validation.json"
      validation = json.loads(validation_path.read_text())
      validation["protocol_sha256"] = "0" * 64
      validation_path.write_text(json.dumps(validation))
      with self.assertRaisesRegex(TopologyError, "protocol artifact drift"):
        observe(fixture)


if __name__ == "__main__":
  unittest.main()
