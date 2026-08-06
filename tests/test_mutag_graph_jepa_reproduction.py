import unittest

from tinygrad import Context, Tensor

from experiments.mutag_graph_jepa_reproduction import Model, PaperBatch, paper_patches
from tinymesh import Graph
from tinymesh.datasets import MUTAG


def dataset() -> MUTAG:
  graphs, nodes, edges, labels = [], [], [], []
  for index in range(10):
    graph = Graph(5, [0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3])
    graphs.append(graph)
    nodes.append(Tensor([index % 7, 1, 2, 3, 4]))
    edges.append(Tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    labels.append(index % 2)
  return MUTAG(tuple(graphs), tuple(nodes), tuple(edges), tuple(labels))


class GraphJEPAReproductionTest(unittest.TestCase):
  def test_paper_partition_preserves_32_slots_and_one_hop_patches(self) -> None:
    import random

    graph = dataset().graphs[0]
    patches = paper_patches(graph, random.Random(0))

    self.assertEqual(len(patches), 32)
    self.assertEqual(sum(bool(patch) for patch in patches), graph.nodes)
    self.assertTrue(patches[-1])
    self.assertTrue(all(1 <= len(patch) <= 3 for patch in patches if patch))

  def test_batch_keeps_empty_slots_explicit(self) -> None:
    batch = PaperBatch(dataset(), (0, 1), seed=0)

    self.assertEqual(batch.mask.shape, (2, 32))
    self.assertEqual(batch.adjacency.shape, (2, 32, 32))
    self.assertEqual(batch.maximum.shape, (2, 32, 15))
    self.assertEqual(batch.mask.sum().item(), 10)

  def test_model_matches_paper_shapes(self) -> None:
    data = dataset()
    batch = PaperBatch(data, (0, 1), seed=0)
    context, targets = batch.selection(1)
    model = Model(7, 4, 16)

    with Context(TRAINING=1):
      prediction, truth = model.predict(batch, context, targets)
    with Context(TRAINING=0):
      embedding = model.embed(batch)

    self.assertEqual(prediction.shape, (2, 3, 2))
    self.assertEqual(truth.shape, (2, 3, 2))
    self.assertEqual(embedding.shape, (2, 16))
    self.assertTrue(all(value == value for value in prediction.tolist()[0][0]))

if __name__ == "__main__":
  unittest.main()
