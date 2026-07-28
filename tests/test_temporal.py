import unittest

from tinygrad import Device, Tensor, dtypes

from tinymesh import Graph, StaticGraphTemporalSignal


def signal() -> StaticGraphTemporalSignal:
    return StaticGraphTemporalSignal(
        Graph(2, [0, 1], [1, 0]),
        ("left", "right"),
        Tensor(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
            ],
            device=Device.DEFAULT,
        ).realize(),
        Tensor(
            [
                [[2.0], [3.0]],
                [[4.0], [5.0]],
                [[6.0], [7.0]],
            ],
            device=Device.DEFAULT,
        ).realize(),
        Tensor.ones(2, device=Device.DEFAULT).realize(),
    )


class TemporalSignalTest(unittest.TestCase):
    def test_indexes_ordered_snapshots(self) -> None:
        dataset = signal()

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.node_ids, ("left", "right"))
        self.assertEqual(dataset[1][0].tolist(), [[3.0], [4.0]])
        self.assertEqual(dataset[-1][1].tolist(), [[6.0], [7.0]])
        self.assertEqual([x[0].item() for x, _ in dataset], [1.0, 3.0, 5.0])
        with self.assertRaises(IndexError):
            _ = dataset[3]

    def test_slices_and_splits_without_copying_topology(self) -> None:
        dataset = signal()
        train, test = dataset.split(2 / 3)

        self.assertEqual((len(train), len(test)), (2, 1))
        self.assertIs(train.graph, dataset.graph)
        self.assertIs(test.graph, dataset.graph)
        self.assertIs(train.edge_weight, dataset.edge_weight)
        self.assertEqual(train.x.tolist(), dataset.x[:2].tolist())
        self.assertEqual(test.x.tolist(), dataset.x[2:].tolist())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _ = dataset[::2]

    def test_rejects_misaligned_fields(self) -> None:
        graph = Graph(2, [0], [1])
        x = Tensor.ones(3, 2, 1, device=Device.DEFAULT)
        y = Tensor.ones(3, 2, 1, device=Device.DEFAULT)

        with self.assertRaisesRegex(ValueError, r"x must have shape \[T, N, F\]"):
            StaticGraphTemporalSignal(graph, ("a", "b"), Tensor.ones(3, 2), y)
        with self.assertRaisesRegex(ValueError, r"y must have shape \[T, N, Y\]"):
            StaticGraphTemporalSignal(graph, ("a", "b"), x, Tensor.ones(3, 2))
        with self.assertRaisesRegex(ValueError, "same time and node axes"):
            StaticGraphTemporalSignal(graph, ("a", "b"), x, Tensor.ones(2, 2, 1))
        with self.assertRaisesRegex(ValueError, "expected 2 node IDs"):
            StaticGraphTemporalSignal(graph, ("a",), x, y)
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            StaticGraphTemporalSignal(graph, ("a", ""), x, y)
        with self.assertRaisesRegex(ValueError, "unique"):
            StaticGraphTemporalSignal(graph, ("a", "a"), x, y)
        with self.assertRaisesRegex(ValueError, r"edge_weight must have shape \[1\]"):
            StaticGraphTemporalSignal(graph, ("a", "b"), x, y, Tensor.ones(2))
        with self.assertRaisesRegex(ValueError, "floating"):
            StaticGraphTemporalSignal(
                graph,
                ("a", "b"),
                x,
                y,
                Tensor.ones(1, dtype=dtypes.int),
            )

    def test_rejects_empty_split(self) -> None:
        dataset = signal()
        for ratio in (0.0, 1.0):
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                dataset.split(ratio)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            dataset[:1].split(0.5)


if __name__ == "__main__":
    unittest.main()
