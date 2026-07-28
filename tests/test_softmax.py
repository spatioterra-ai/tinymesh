import unittest
from math import exp, prod

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.uop.ops import Ops

from tinymesh import Graph


SOURCE = [1, 0, 1, 3, 1, 0]
TARGET = [2, 1, 2, 2, 4, 2]
SCORE = [2.0, -1.0, 3.0, 4.0, -2.0, 5.0]
GRADIENT = [1.0, 2.0, 5.0, 7.0, 11.0, 13.0]


def reference(target=TARGET, score=SCORE, gradient=GRADIENT):
    maximum = [float("-inf")] * 6
    for edge_target, value in zip(target, score):
        maximum[edge_target] = max(maximum[edge_target], value)
    exponential = [exp(value - maximum[edge_target]) for edge_target, value in zip(target, score)]
    total = [0.0] * 6
    for edge_target, value in zip(target, exponential):
        total[edge_target] += value
    attention = [value / total[edge_target] for edge_target, value in zip(target, exponential)]
    mean_gradient = [0.0] * 6
    for edge_target, weight, value in zip(target, attention, gradient):
        mean_gradient[edge_target] += weight * value
    score_gradient = [
        weight * (value - mean_gradient[edge_target])
        for edge_target, weight, value in zip(target, attention, gradient)
    ]
    return attention, score_gradient


def run(source=SOURCE, target=TARGET, score=SCORE, gradient=GRADIENT):
    graph = Graph(6, source, target)
    edge_score = Tensor(score, device=Device.DEFAULT).realize()
    attention = graph.softmax(edge_score)
    score_gradient = attention.gradient(
        edge_score,
        gradient=Tensor(gradient, device=Device.DEFAULT),
    )[0]
    Tensor.realize(attention, score_gradient)
    return attention.tolist(), score_gradient.tolist()


class GraphSoftmaxTest(unittest.TestCase):
    def test_forward_and_gradient_match_reference(self):
        actual_attention, actual_gradient = run()
        expected_attention, expected_gradient = reference()

        for actual, expected in zip(actual_attention, expected_attention):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(actual_gradient, expected_gradient):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_is_stable_and_normalized_by_target(self):
        graph = Graph(5, [0, 1, 2, 3], [4, 4, 4, 3])
        attention = graph.softmax(
            Tensor([10_000.0, 10_001.0, 9_999.0, -10_000.0], device=Device.DEFAULT)
        ).tolist()

        self.assertAlmostEqual(sum(attention[:3]), 1.0, places=6)
        self.assertEqual(attention[3], 1.0)
        self.assertGreater(attention[1], attention[0])
        self.assertGreater(attention[0], attention[2])

    def test_coo_order_moves_scores_and_gradients(self):
        actual = run(
            list(reversed(SOURCE)),
            list(reversed(TARGET)),
            list(reversed(SCORE)),
            list(reversed(GRADIENT)),
        )
        expected = reference(
            list(reversed(TARGET)),
            list(reversed(SCORE)),
            list(reversed(GRADIENT)),
        )

        for actual_values, expected_values in zip(actual, expected):
            for actual_value, expected_value in zip(actual_values, expected_values):
                self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_empty_graph(self):
        graph = Graph(3, [], [])
        score = Tensor([], device=Device.DEFAULT).realize()
        attention = graph.softmax(score)
        gradient = attention.sum().gradient(score)[0]

        self.assertEqual(attention.tolist(), [])
        self.assertEqual(gradient.tolist(), [])

    def test_single_node(self):
        graph = Graph(1, [0, 0, 0], [0, 0, 0])
        score = Tensor([1.0, 2.0, 3.0], device=Device.DEFAULT).realize()
        attention = graph.softmax(score)
        gradient = attention.gradient(
            score,
            gradient=Tensor([1.0, 2.0, 4.0], device=Device.DEFAULT),
        )[0]
        expected_attention, expected_gradient = reference(
            [0, 0, 0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
        )

        for actual, expected in zip(attention.tolist(), expected_attention):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(gradient.tolist(), expected_gradient):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_rejects_incompatible_scores(self):
        graph = Graph(2, [0], [1])
        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            graph.softmax(Tensor.ones(1, 1, device=Device.DEFAULT))
        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            graph.softmax(Tensor.ones(2, device=Device.DEFAULT))
        with self.assertRaisesRegex(ValueError, "floating"):
            graph.softmax(Tensor([1], dtype=dtypes.int32, device=Device.DEFAULT))

    def test_forward_and_gradient_have_sparse_structure(self):
        nodes, edges = 5, 7
        source = [0, 1, 1, 2, 3, 4, 4]
        target = [1, 0, 3, 3, 3, 0, 3]
        graph = Graph(nodes, source, target)
        score = Tensor.ones(edges, device=Device.DEFAULT).realize()
        attention = graph.softmax(score)
        gradient = attention.gradient(
            score,
            gradient=Tensor(list(range(edges)), device=Device.DEFAULT),
        )[0]

        for tensor in (attention, gradient):
            calls = [uop for uop in tensor.uop.toposort() if uop.op is Ops.CALL]
            self.assertLessEqual(len(calls), 6)
            for call in calls:
                self._assert_sparse_kernel(call.src[0], nodes=nodes, edges=edges)
            self.assertTrue(all(
                prod(int(size) for size in uop._shape) <= max(nodes + 1, edges)
                for uop in tensor.uop.toposort()
                if uop._shape is not None
            ))

    def _assert_sparse_kernel(self, body: UOp, *, nodes: int, edges: int) -> None:
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        static = [uop for uop in ranges if uop.src[0].op is Ops.CONST]
        self.assertEqual(len(static), 1)
        self.assertLessEqual(int(static[0].src[0]), max(nodes, edges))
        self.assertLessEqual(len(ranges), 2)
        for dynamic in set(ranges) - set(static):
            self.assertEqual(dynamic.dtype, UOp.loop(-1).dtype)


if __name__ == "__main__":
    unittest.main()
