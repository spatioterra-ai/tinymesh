import unittest

from tinygrad import Device, Tensor

from experiments.mutag_jepa import GraphBatch, _folds
from tinymesh import Graph
from tinymesh.datasets import MUTAG


def dataset() -> MUTAG:
    graphs, node_labels, edge_labels, labels = [], [], [], []
    for index in range(10):
        graphs.append(Graph(3, [0, 1, 1, 2], [1, 0, 2, 1]))
        node_labels.append(Tensor([index % 2, 1, 2], device=Device.DEFAULT).realize())
        edge_labels.append(Tensor([0, 0, 1, 1], device=Device.DEFAULT).realize())
        labels.append(index % 2)
    return MUTAG(tuple(graphs), tuple(node_labels), tuple(edge_labels), tuple(labels))


class MUTAGJEPATest(unittest.TestCase):
    def test_batch_keeps_graphs_disjoint_and_pools_sparsely(self) -> None:
        batch = GraphBatch(dataset(), (1, 3), mask_every=2, seed=0)
        values = Tensor([[float(index)] for index in range(6)], device=Device.DEFAULT).realize()

        self.assertEqual(batch.graph.nodes, 6)
        self.assertEqual(batch.graph.source, (0, 1, 1, 2, 3, 4, 4, 5))
        self.assertEqual(batch.pool_graph.edges, 6)
        self.assertEqual(batch.context.shape, (6, 8))
        self.assertEqual(batch.target.shape, (6, 8))
        self.assertEqual(batch.pool(values).tolist(), [[1.0], [4.0]])

    def test_stratified_folds_are_disjoint_and_exhaustive(self) -> None:
        labels = dataset().labels
        partitions = _folds(labels, 5, seed=0)

        self.assertEqual([len(partition) for partition in partitions], [2] * 5)
        self.assertEqual(sorted(index for partition in partitions for index in partition), list(range(10)))
        self.assertTrue(all(sorted(labels[index] for index in partition) == [0, 1] for partition in partitions))
        self.assertEqual(partitions, _folds(labels, 5, seed=0))
        self.assertNotEqual(partitions, _folds(labels, 5, seed=1))


if __name__ == "__main__":
    unittest.main()
