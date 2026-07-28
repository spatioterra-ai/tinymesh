import unittest
from math import prod

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, Ops

from tinymesh import Graph


SOURCE = [1, 0, 1, 3, 1, 0]
TARGET = [2, 1, 2, 2, 4, 2]
WEIGHT = [2.0, -1.0, 3.0, 4.0, -2.0, 5.0]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0], [19.0, 23.0], [29.0, 31.0]]
GRADIENT = [[1.0, 10.0], [2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0], [23.0, 29.0]]


def reference(source=SOURCE, target=TARGET, weight=WEIGHT, values=VALUES, gradient=GRADIENT):
    output = [[0.0, 0.0] for _ in values]
    values_gradient = [[0.0, 0.0] for _ in values]
    weight_gradient = []
    for edge_source, edge_target, edge_weight in zip(source, target, weight):
        for feature in range(2):
            output[edge_target][feature] += edge_weight * values[edge_source][feature]
            values_gradient[edge_source][feature] += edge_weight * gradient[edge_target][feature]
        weight_gradient.append(sum(
            values[edge_source][feature] * gradient[edge_target][feature]
            for feature in range(2)
        ))
    return output, values_gradient, weight_gradient


def run(source=SOURCE, target=TARGET, weight=WEIGHT):
    graph = Graph(6, source, target)
    values = Tensor(VALUES, device=Device.DEFAULT).realize()
    edge_weight = Tensor(weight, device=Device.DEFAULT).realize()
    output = graph.sum(values, edge_weight=edge_weight)
    gradient = Tensor(GRADIENT, device=Device.DEFAULT).realize()
    values_gradient, weight_gradient = output.gradient(values, edge_weight, gradient=gradient)
    Tensor.realize(output, values_gradient, weight_gradient)
    return output.tolist(), values_gradient.tolist(), weight_gradient.tolist()


class WeightedAggregationTest(unittest.TestCase):
    def test_forward_and_both_gradients_match_reference(self):
        self.assertEqual(run(), reference())

    def test_shared_weights_accumulate_gradients_across_batches(self):
        values_data = [VALUES, [[2 * value for value in row] for row in VALUES]]
        gradient_data = [GRADIENT, [[0.5 * value for value in row] for row in GRADIENT]]
        graph = Graph(6, SOURCE, TARGET)
        values = Tensor(values_data, device=Device.DEFAULT).realize()
        edge_weight = Tensor(WEIGHT, device=Device.DEFAULT).realize()
        output = graph.sum(values, edge_weight=edge_weight)
        values_gradient, weight_gradient = output.gradient(
            values,
            edge_weight,
            gradient=Tensor(gradient_data, device=Device.DEFAULT),
        )
        expected = [
            reference(values=lane_values, gradient=lane_gradient)
            for lane_values, lane_gradient in zip(values_data, gradient_data)
        ]

        self.assertEqual(output.tolist(), [lane[0] for lane in expected])
        self.assertEqual(values_gradient.tolist(), [lane[1] for lane in expected])
        self.assertEqual(
            weight_gradient.tolist(),
            [sum(lane[2][edge] for lane in expected) for edge in range(len(WEIGHT))],
        )

    def test_edge_order_moves_weights_and_gradients_with_connectivity(self):
        actual = run(list(reversed(SOURCE)), list(reversed(TARGET)), list(reversed(WEIGHT)))
        expected = reference(list(reversed(SOURCE)), list(reversed(TARGET)), list(reversed(WEIGHT)))

        self.assertEqual(actual, expected)
        self.assertEqual(actual[:2], reference()[:2])
        self.assertEqual(list(reversed(actual[2])), reference()[2])

    def test_duplicate_edges_keep_distinct_weight_slots(self):
        graph = Graph(6, SOURCE, TARGET)
        edge_weight = Tensor(WEIGHT, device=Device.DEFAULT).realize()
        values = Tensor(VALUES, device=Device.DEFAULT).realize()
        output = graph.sum(values, edge_weight=edge_weight)
        weight_gradient = output.gradient(
            edge_weight,
            gradient=Tensor(GRADIENT, device=Device.DEFAULT),
        )[0]

        self.assertEqual([edge_weight.tolist()[edge] for edge in (0, 2)], [2.0, 3.0])
        self.assertEqual([weight_gradient.tolist()[edge] for edge in (0, 2)], [50.0, 50.0])

    def test_empty_graph(self):
        graph = Graph(3, [], [])
        values = Tensor([[1.0], [2.0], [3.0]], device=Device.DEFAULT).realize()
        edge_weight = Tensor([], device=Device.DEFAULT).realize()
        output = graph.sum(values, edge_weight=edge_weight)
        values_gradient, weight_gradient = output.sum().gradient(values, edge_weight)

        self.assertEqual(output.tolist(), [[0.0], [0.0], [0.0]])
        self.assertEqual(values_gradient.tolist(), [[0.0], [0.0], [0.0]])
        self.assertEqual(weight_gradient.tolist(), [])

    def test_single_node(self):
        graph = Graph(1, [0, 0, 0], [0, 0, 0])
        values = Tensor([[2.0, 3.0]], device=Device.DEFAULT).realize()
        edge_weight = Tensor([2.0, -1.0, 4.0], device=Device.DEFAULT).realize()
        output = graph.sum(values, edge_weight=edge_weight)
        values_gradient, weight_gradient = output.gradient(
            values,
            edge_weight,
            gradient=Tensor([[5.0, 7.0]], device=Device.DEFAULT),
        )

        self.assertEqual(output.tolist(), [[10.0, 15.0]])
        self.assertEqual(values_gradient.tolist(), [[25.0, 35.0]])
        self.assertEqual(weight_gradient.tolist(), [31.0, 31.0, 31.0])

    def test_rejects_incompatible_weight_tensors(self):
        graph = Graph(2, [0], [1])
        values = Tensor.ones(2, 1, device=Device.DEFAULT)

        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            graph.sum(values, edge_weight=Tensor.ones(1, 1, device=Device.DEFAULT))
        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            graph.sum(values, edge_weight=Tensor.ones(2, device=Device.DEFAULT))
        with self.assertRaisesRegex(ValueError, "same dtype"):
            graph.sum(values, edge_weight=Tensor([1], dtype=dtypes.int32, device=Device.DEFAULT))
        other_device = "PYTHON" if Device.DEFAULT != "PYTHON" else "CPU"
        with self.assertRaisesRegex(ValueError, "one shared device"):
            graph.sum(values, edge_weight=Tensor([1.0], device=other_device))

    def test_forward_and_gradients_have_sparse_structure(self):
        source = [0, 1, 1, 2, 3, 4, 4]
        target = [1, 0, 3, 3, 3, 0, 3]
        graph = Graph(5, source, target)
        values = Tensor.ones(5, 3, device=Device.DEFAULT).realize()
        edge_weight = Tensor.ones(7, device=Device.DEFAULT).realize()
        output = graph.sum(values, edge_weight=edge_weight)
        values_gradient, weight_gradient = output.gradient(
            values,
            edge_weight,
            gradient=Tensor.ones(5, 3, device=Device.DEFAULT),
        )

        self._assert_csr_kernel(output, nodes=5, edges=7, width=3)
        self._assert_csr_kernel(values_gradient, nodes=5, edges=7, width=3)
        self._assert_edge_kernel(weight_gradient, nodes=5, edges=7, width=3)

    def _assert_csr_kernel(self, tensor, *, nodes, edges, width):
        body = self._kernel_body(tensor)
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        self.assertEqual([uop.arg[1] for uop in ranges], [AxisType.LOOP, AxisType.LOOP])
        self.assertEqual(ranges[0].src[0].arg, nodes * width)
        self.assertEqual(ranges[1].dtype, UOp.loop(-1).dtype)
        self._assert_no_edge_feature_state(body, nodes, edges, width)

    def _assert_edge_kernel(self, tensor, *, nodes, edges, width):
        body = self._kernel_body(tensor)
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        self.assertEqual([uop.arg[1] for uop in ranges], [AxisType.LOOP, AxisType.REDUCE])
        self.assertEqual([uop.src[0].arg for uop in ranges], [edges, width])
        self._assert_no_edge_feature_state(body, nodes, edges, width)

    def _kernel_body(self, tensor):
        calls = [uop for uop in tensor.uop.toposort() if uop.op is Ops.CALL]
        self.assertEqual(len(calls), 1)
        return calls[0].src[0]

    def _assert_no_edge_feature_state(self, body, nodes, edges, width):
        limit = max(nodes * width, edges, nodes + 1)
        self.assertTrue(all(
            prod(int(size) for size in uop._shape) <= limit
            for uop in body.toposort()
            if uop._shape is not None
        ))


if __name__ == "__main__":
    unittest.main()
