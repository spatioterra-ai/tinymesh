import unittest
from math import prod

from tinygrad import Device, Tensor, UOp
from tinygrad.uop.ops import Ops

from tinymesh import Graph


SOURCE = [1, 0, 1, 3, 1, 0]
TARGET = [2, 1, 2, 2, 4, 2]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0], [19.0, 23.0], [29.0, 31.0]]
EDGE_GRADIENT = [[1.0, 10.0], [2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0], [23.0, 29.0]]


def reference(index):
    output = [VALUES[node] for node in index]
    gradient = [[0.0, 0.0] for _ in VALUES]
    for node, edge_gradient in zip(index, EDGE_GRADIENT):
        for feature, value in enumerate(edge_gradient):
            gradient[node][feature] += value
    return output, gradient


def run(source=SOURCE, target=TARGET, endpoint="source"):
    graph = Graph(6, source, target)
    values = Tensor(VALUES, device=Device.DEFAULT).realize()
    output = graph.edge_values(values, endpoint=endpoint)
    gradient = output.gradient(
        values,
        gradient=Tensor(EDGE_GRADIENT, device=Device.DEFAULT),
    )[0]
    Tensor.realize(output, gradient)
    return output.tolist(), gradient.tolist()


class EdgeValuesTest(unittest.TestCase):
    def test_source_and_target_values_and_gradients(self):
        for endpoint, index in (("source", SOURCE), ("target", TARGET)):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(run(endpoint=endpoint), reference(index))

    def test_coo_order_is_preserved(self):
        for endpoint, index in (("source", SOURCE), ("target", TARGET)):
            with self.subTest(endpoint=endpoint):
                actual = run(
                    list(reversed(SOURCE)),
                    list(reversed(TARGET)),
                    endpoint,
                )
                expected_output, expected_gradient = reference(list(reversed(index)))
                self.assertEqual(actual, (expected_output, expected_gradient))

    def test_empty_graph(self):
        graph = Graph(3, [], [])
        values = Tensor([[1.0], [2.0], [3.0]], device=Device.DEFAULT).realize()
        output = graph.edge_values(values, endpoint="source")
        gradient = output.sum().gradient(values)[0]

        self.assertEqual(output.shape, (0, 1))
        self.assertEqual(gradient.tolist(), [[0.0], [0.0], [0.0]])

    def test_single_node(self):
        graph = Graph(1, [0, 0, 0], [0, 0, 0])
        values = Tensor([[2.0, 3.0]], device=Device.DEFAULT).realize()
        output = graph.edge_values(values, endpoint="target")
        gradient = output.sum().gradient(values)[0]

        self.assertEqual(output.tolist(), [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        self.assertEqual(gradient.tolist(), [[3.0, 3.0]])

    def test_rejects_unknown_endpoint(self):
        graph = Graph(2, [0], [1])
        with self.assertRaisesRegex(ValueError, "source.*target"):
            graph.edge_values(Tensor.ones(2, 1), endpoint="edge")  # type: ignore[arg-type]

    def test_rejects_batch_axes(self):
        graph = Graph(2, [0], [1])
        with self.assertRaisesRegex(ValueError, r"shape \[N, H\]"):
            graph.edge_values(Tensor.ones(3, 2, 1), endpoint="source")

    def test_forward_and_backward_have_sparse_structure(self):
        source = [0, 1, 1, 2, 3, 4, 4]
        target = [1, 0, 3, 3, 3, 0, 3]
        graph = Graph(5, source, target)
        values = Tensor.ones(5, 3, device=Device.DEFAULT).realize()
        output = graph.edge_values(values, endpoint="source")
        gradient = output.gradient(
            values,
            gradient=Tensor.ones(7, 3, device=Device.DEFAULT),
        )[0]

        self._assert_edge_kernel(output, edges=7, width=3)
        self._assert_csr_kernel(gradient, nodes=5, edges=7, width=3)

    def _assert_edge_kernel(self, tensor: Tensor, *, edges: int, width: int) -> None:
        body = self._kernel_body(tensor)
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        self.assertEqual(len(ranges), 1)
        self.assertEqual(int(ranges[0].src[0]), edges * width)
        self._assert_sparse_shapes(body, nodes=0, edges=edges, width=width)

    def _assert_csr_kernel(self, tensor: Tensor, *, nodes: int, edges: int, width: int) -> None:
        body = self._kernel_body(tensor)
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        self.assertEqual(len(ranges), 2)
        self.assertEqual(int(ranges[0].src[0]), nodes * width)
        self.assertEqual(ranges[1].dtype, UOp.loop(-1).dtype)
        self._assert_sparse_shapes(body, nodes=nodes, edges=edges, width=width)

    def _kernel_body(self, tensor: Tensor) -> UOp:
        calls = [uop for uop in tensor.uop.toposort() if uop.op is Ops.CALL]
        self.assertEqual(len(calls), 1)
        return calls[0].src[0]

    def _assert_sparse_shapes(self, body: UOp, *, nodes: int, edges: int, width: int) -> None:
        limit = max(nodes * width, edges * width, nodes + 1)
        self.assertTrue(all(
            prod(int(size) for size in uop._shape) <= limit
            for uop in body.toposort()
            if uop._shape is not None
        ))


if __name__ == "__main__":
    unittest.main()
