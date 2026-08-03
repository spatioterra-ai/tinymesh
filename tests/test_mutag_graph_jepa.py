import unittest

from tinygrad import Device, Tensor

from experiments.mutag_graph_jepa import ARMS, Model, PatchBatch, _objective, _partition, _patches, _rwse
from tinymesh import Graph
from tinymesh.datasets import MUTAG


def dataset() -> MUTAG:
  graphs, node_labels, edge_labels, labels = [], [], [], []
  for index in range(10):
    source = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    target = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    graphs.append(Graph(6, source, target))
    node_labels.append(Tensor([(node + index) % 3 for node in range(6)], device=Device.DEFAULT).realize())
    edge_labels.append(Tensor([edge % 2 for edge in range(10)], device=Device.DEFAULT).realize())
    labels.append(index % 2)
  return MUTAG(tuple(graphs), tuple(node_labels), tuple(edge_labels), tuple(labels))


class GraphJEPAPatchTest(unittest.TestCase):
  def test_partition_is_balanced_disjoint_and_deterministic(self) -> None:
    first = _partition(7, 3, seed=2)

    self.assertEqual([len(partition) for partition in first], [3, 2, 2])
    self.assertEqual(sorted(node for partition in first for node in partition), list(range(7)))
    self.assertEqual(first, _partition(7, 3, seed=2))
    self.assertNotEqual(first, _partition(7, 3, seed=3))

  def test_patch_adds_one_hop_without_losing_its_base(self) -> None:
    graph = dataset().graphs[0]
    bases = _partition(graph.nodes, 3, seed=0)
    patches = _patches(graph, 3, seed=0)

    self.assertEqual(len(patches), 3)
    self.assertTrue(all(set(base) <= set(patch) for base, patch in zip(bases, patches)))
    self.assertTrue(all(set(patch) <= set(range(graph.nodes)) for patch in patches))
    self.assertTrue(any(len(patch) > len(base) for base, patch in zip(bases, patches)))

  def test_rwse_is_exact_without_node_pair_storage(self) -> None:
    edge = Graph(2, [0, 1], [1, 0])
    triangle = Graph(3, [0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2])

    self.assertEqual(_rwse(edge, 4), ((0.0, 1.0, 0.0, 1.0),) * 2)
    for row in _rwse(triangle, 3):
      self.assertEqual(row, (0.0, 0.5, 0.25))

  def test_batch_keeps_patch_and_graph_ownership_explicit(self) -> None:
    batch = PatchBatch(dataset(), tuple(range(10)), patches=3, walk_length=4, seed=0)

    self.assertEqual(batch.graphs, 10)
    self.assertEqual(batch.graph_pool.items, 30)
    self.assertEqual(batch.graph_pool.owners, tuple(graph for graph in range(10) for _ in range(3)))
    self.assertEqual(batch.nodes.shape, (batch.node_pool.items, 7))
    self.assertEqual(batch.edges.shape, (batch.graph.edges, 4))
    self.assertEqual(batch.position.shape, (30, 4))
    self.assertEqual(batch.node_pool(Tensor.ones(batch.node_pool.items, 1)).tolist(), [[1.0]] * 30)

  def test_selection_is_within_graph_and_disjoint(self) -> None:
    batch = PatchBatch(dataset(), tuple(range(10)), patches=3, walk_length=2, seed=0)
    context, targets = batch.selection(seed=4, targets=2)

    for graph, (context_patch, target_patches) in enumerate(zip(context.tolist(), targets.tolist())):
      self.assertEqual(context_patch // 3, graph)
      self.assertTrue(all(target // 3 == graph for target in target_patches))
      self.assertEqual(len({context_patch, *target_patches}), 3)
    self.assertEqual((context.tolist(), targets.tolist()), tuple(value.tolist() for value in batch.selection(4, 2)))

  def test_model_predicts_one_code_per_target(self) -> None:
    data = dataset()
    batch = PatchBatch(data, tuple(range(10)), patches=3, walk_length=2, seed=0)
    context, targets = batch.selection(seed=0, targets=2)
    model = Model(len(data.node_types), len(data.bond_types), 2, 4, ARMS[1])
    prediction, target = model.predict(batch, context, targets)

    self.assertEqual(prediction.shape, (10, 2, 2))
    self.assertEqual(target.shape, prediction.shape)
    self.assertGreaterEqual(_objective(prediction, target, "smooth_l1").item(), 0)


if __name__ == "__main__":
  unittest.main()
