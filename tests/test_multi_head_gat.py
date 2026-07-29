import unittest
from math import exp

from tinygrad import Device, Tensor

from experiments.multi_head_gat import fit_one_step
from tinymesh import Graph
from tinymesh.nn import GATConv


SOURCE = [0, 1, 2, 0, 1]
TARGET = [0, 0, 1, 2, 2]
VALUES = [1.0, 3.0, 7.0]
WEIGHT = [2.0, -1.0]
SOURCE_ATTENTION = [0.5, -1.0]
TARGET_ATTENTION = [1.5, 0.25]


def reference(
    weight=WEIGHT,
    source_attention=SOURCE_ATTENTION,
    target_attention=TARGET_ATTENTION,
):
    state = [[value * head_weight for head_weight in weight] for value in VALUES]
    score = [
        [
            state[edge_source][head] * source_attention[head]
            + state[edge_target][head] * target_attention[head]
            for head in range(2)
        ]
        for edge_source, edge_target in zip(SOURCE, TARGET)
    ]
    score = [[value if value >= 0 else 0.2 * value for value in edge] for edge in score]
    output = [[0.0, 0.0] for _ in VALUES]
    for head in range(2):
        for node in range(len(VALUES)):
            edges = [edge for edge, target in enumerate(TARGET) if target == node]
            if not edges:
                continue
            maximum = max(score[edge][head] for edge in edges)
            exponential = [exp(score[edge][head] - maximum) for edge in edges]
            total = sum(exponential)
            for edge, numerator in zip(edges, exponential):
                output[node][head] += numerator / total * state[SOURCE[edge]][head]
    return output


def run(
    weight=WEIGHT,
    source_attention=SOURCE_ATTENTION,
    target_attention=TARGET_ATTENTION,
):
    model = GATConv(1, 1, heads=2, bias=False)
    model.linear.weight = Tensor([[value] for value in weight], device=Device.DEFAULT).realize()
    model.source_attention = Tensor([[value] for value in source_attention], device=Device.DEFAULT).realize()
    model.target_attention = Tensor([[value] for value in target_attention], device=Device.DEFAULT).realize()
    return model(
        Tensor([[value] for value in VALUES], device=Device.DEFAULT).realize(),
        Graph(3, SOURCE, TARGET),
    ).tolist()


class MultiHeadGATTest(unittest.TestCase):
    def test_matches_reference(self):
        for actual_row, expected_row in zip(run(), reference()):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_head_permutation_only_moves_output_columns(self):
        expected = run()
        actual = run(
            list(reversed(WEIGHT)),
            list(reversed(SOURCE_ATTENTION)),
            list(reversed(TARGET_ATTENTION)),
        )

        for actual_row, expected_row in zip(actual, expected):
            self.assertAlmostEqual(actual_row[0], expected_row[1], places=5)
            self.assertAlmostEqual(actual_row[1], expected_row[0], places=5)

    def test_optimizer_updates_both_heads(self):
        observation = fit_one_step(Device.DEFAULT)

        self.assertAlmostEqual(observation.initial_loss, 0.214323, places=5)
        self.assertAlmostEqual(observation.source_attention_gradient[0], -0.197655, places=5)
        self.assertAlmostEqual(observation.source_attention_gradient[1], 0.197655, places=5)
        self.assertLess(observation.final_loss, observation.initial_loss)
        self.assertGreater(observation.source_attention[0], 1.0)
        self.assertLess(observation.source_attention[1], -1.0)

    def test_rejects_non_positive_head_count(self):
        with self.assertRaisesRegex(ValueError, "head"):
            GATConv(1, 1, heads=0)


if __name__ == "__main__":
    unittest.main()
