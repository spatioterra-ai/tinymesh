import unittest
from math import sqrt

from tinygrad import Device, Tensor

from experiments.gcn import fit_one_step
from tinymesh import Graph
from tinymesh.nn import GCNConv


SOURCE = [0, 1, 2, 0, 1, 1, 2]
TARGET = [0, 1, 2, 1, 0, 2, 1]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]
WEIGHT = [[2.0, -1.0], [1.0, 3.0]]


def dense_gcn(
    nodes: int,
    source: list[int],
    target: list[int],
    values: list[list[float]],
    weight: list[list[float]],
) -> list[list[float]]:
    degree = [0] * nodes
    for node in target:
        degree[node] += 1
    transformed = [
        [sum(feature * coefficient for feature, coefficient in zip(row, output)) for output in weight]
        for row in values
    ]
    result = [[0.0] * len(weight) for _ in range(nodes)]
    for source_node, target_node in zip(source, target):
        scale = 1.0 / sqrt(degree[source_node] * degree[target_node])
        for feature, value in enumerate(transformed[source_node]):
            result[target_node][feature] += value * scale
    return result


def run(
    source: list[int] = SOURCE,
    target: list[int] = TARGET,
    values: list[list[float]] = VALUES,
) -> list[list[float]]:
    model = GCNConv(2, 2, bias=False)
    model.linear.weight = Tensor(WEIGHT, device=Device.DEFAULT).realize()
    return model(
        Tensor(values, device=Device.DEFAULT).realize(),
        Graph(3, source, target),
    ).tolist()


class GCNTest(unittest.TestCase):
    def test_matches_dense_reference(self) -> None:
        expected = dense_gcn(3, SOURCE, TARGET, VALUES, WEIGHT)

        for actual_row, expected_row in zip(run(), expected):
            for actual, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected_value, places=5)

    def test_is_vertex_permutation_equivariant(self) -> None:
        old_to_new = [2, 0, 1]
        values = [[0.0, 0.0] for _ in VALUES]
        for old, new in enumerate(old_to_new):
            values[new] = VALUES[old]

        expected = run()
        actual = run(
            [old_to_new[node] for node in SOURCE],
            [old_to_new[node] for node in TARGET],
            values,
        )

        for old, new in enumerate(old_to_new):
            for actual_value, expected_value in zip(actual[new], expected[old]):
                self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_explicit_self_loop_retains_disconnected_node(self) -> None:
        model = GCNConv(1, 1, bias=False)
        model.linear.weight = Tensor([[2.0]], device=Device.DEFAULT).realize()
        graph = Graph(3, [0, 1, 2, 0, 1], [0, 1, 2, 1, 0])
        values = Tensor([[1.0], [3.0], [5.0]], device=Device.DEFAULT).realize()

        for actual, expected in zip(model(values, graph).tolist(), [4.0, 4.0, 10.0]):
            self.assertAlmostEqual(actual[0], expected, places=5)

    def test_zero_degree_node_returns_zero(self) -> None:
        model = GCNConv(1, 1, bias=False)
        model.linear.weight = Tensor([[2.0]], device=Device.DEFAULT).realize()
        graph = Graph(3, [0, 1, 0, 1], [0, 1, 1, 0])
        values = Tensor([[1.0], [3.0], [5.0]], device=Device.DEFAULT).realize()

        self.assertEqual(model(values, graph).tolist()[2], [0.0])

    def test_bias_is_applied_after_propagation(self) -> None:
        model = GCNConv(1, 1)
        model.linear.weight = Tensor.zeros(1, 1, device=Device.DEFAULT).realize()
        model.linear.bias = Tensor([2.0], device=Device.DEFAULT).realize()

        self.assertEqual(
            model(Tensor.zeros(2, 1), Graph(2, [], [])).tolist(),
            [[2.0], [2.0]],
        )

    def test_batch_axis_matches_independent_graphs(self) -> None:
        model = GCNConv(2, 2, bias=False)
        model.linear.weight = Tensor(WEIGHT, device=Device.DEFAULT).realize()
        graph = Graph(3, SOURCE, TARGET)
        values = Tensor(
            [VALUES, [[2 * value for value in row] for row in VALUES]],
            device=Device.DEFAULT,
        ).realize()

        expected = Tensor.stack(*(model(lane, graph) for lane in values))
        self.assertEqual(model(values, graph).tolist(), expected.tolist())

    def test_optimizer_updates_parameter_from_normalized_sum(self) -> None:
        observation = fit_one_step(Device.DEFAULT)

        self.assertAlmostEqual(observation.initial_loss, 1.0)
        self.assertAlmostEqual(observation.weight_gradient, -1.0, places=5)
        self.assertAlmostEqual(observation.final_loss, 0.0, places=5)
        self.assertAlmostEqual(observation.weight, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
