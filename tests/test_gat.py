import unittest
from math import exp

from tinygrad import Device, Tensor

from experiments.gat import fit_one_step
from tinymesh import Graph
from tinymesh.nn import GATConv


SOURCE = [0, 1, 2, 0, 1]
TARGET = [0, 0, 1, 2, 2]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]
WEIGHT = [[2.0, -1.0], [1.0, 3.0]]
SOURCE_ATTENTION = [0.5, -1.0]
TARGET_ATTENTION = [1.5, 0.25]


def reference(source=SOURCE, target=TARGET, values=VALUES):
    state = [
        [sum(value * coefficient for value, coefficient in zip(row, output)) for output in WEIGHT]
        for row in values
    ]
    source_score = [sum(value * coefficient for value, coefficient in zip(row, SOURCE_ATTENTION)) for row in state]
    target_score = [sum(value * coefficient for value, coefficient in zip(row, TARGET_ATTENTION)) for row in state]
    score = [
        source_score[edge_source] + target_score[edge_target]
        for edge_source, edge_target in zip(source, target)
    ]
    score = [value if value >= 0 else 0.2 * value for value in score]
    maximum = [float("-inf")] * len(values)
    for edge_target, value in zip(target, score):
        maximum[edge_target] = max(maximum[edge_target], value)
    exponential = [exp(value - maximum[edge_target]) for edge_target, value in zip(target, score)]
    total = [0.0] * len(values)
    for edge_target, value in zip(target, exponential):
        total[edge_target] += value
    attention = [value / total[edge_target] for edge_target, value in zip(target, exponential)]
    output = [[0.0] * len(WEIGHT) for _ in values]
    for edge_source, edge_target, edge_attention in zip(source, target, attention):
        for feature, value in enumerate(state[edge_source]):
            output[edge_target][feature] += edge_attention * value
    return output


def run(source=SOURCE, target=TARGET, values=VALUES):
    model = GATConv(2, 2, bias=False)
    model.linear.weight = Tensor(WEIGHT, device=Device.DEFAULT).realize()
    model.source_attention = Tensor([SOURCE_ATTENTION], device=Device.DEFAULT).realize()
    model.target_attention = Tensor([TARGET_ATTENTION], device=Device.DEFAULT).realize()
    return model(
        Tensor(values, device=Device.DEFAULT).realize(),
        Graph(3, source, target),
    ).tolist()


class GATTest(unittest.TestCase):
    def test_matches_reference(self):
        for actual_row, expected_row in zip(run(), reference()):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_is_vertex_permutation_equivariant(self):
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

    def test_bias_is_applied_after_head_concatenation(self):
        model = GATConv(1, 1, heads=2)
        model.linear.weight = Tensor.zeros(2, 1, device=Device.DEFAULT).realize()
        model.bias = Tensor([2.0, 3.0], device=Device.DEFAULT).realize()

        self.assertEqual(
            model(Tensor.zeros(2, 1), Graph(2, [], [])).tolist(),
            [[2.0, 3.0], [2.0, 3.0]],
        )

    def test_optimizer_updates_attention_through_sparse_softmax(self):
        observation = fit_one_step(Device.DEFAULT)

        self.assertAlmostEqual(observation.initial_loss, 0.214323, places=5)
        self.assertAlmostEqual(observation.source_attention_gradient, -0.395310, places=5)
        self.assertAlmostEqual(observation.final_loss, 0.126819, places=5)
        self.assertAlmostEqual(observation.linear_weight, 1.089257, places=5)
        self.assertAlmostEqual(observation.source_attention, 1.039531, places=5)
        self.assertEqual(observation.target_attention, 0.0)


if __name__ == "__main__":
    unittest.main()
