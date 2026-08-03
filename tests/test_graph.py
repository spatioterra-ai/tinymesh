import unittest
from dataclasses import FrozenInstanceError
from math import prod

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.uop.ops import Ops

from tinymesh import Graph


SOURCE = [1, 0, 1, 3, 1, 0]
TARGET = [2, 1, 2, 2, 4, 2]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0], [19.0, 23.0], [29.0, 31.0]]
GRADIENT = [[1.0, 10.0], [2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0], [23.0, 29.0]]
EXPECTED = [[0.0, 0.0], [1.0, 2.0], [20.0, 29.0], [0.0, 0.0], [3.0, 5.0], [0.0, 0.0]]
EXPECTED_GRADIENT = [[7.0, 10.0], [27.0, 33.0], [0.0, 0.0], [5.0, 7.0], [0.0, 0.0], [0.0, 0.0]]


def run(source=SOURCE, target=TARGET, values=VALUES, gradient_values=GRADIENT):
    graph = Graph(6, source, target)
    state = Tensor(values, device=Device.DEFAULT).realize()
    output = graph.sum(state)
    gradient = Tensor(gradient_values, device=Device.DEFAULT).realize()
    state_gradient = output.gradient(state, gradient=gradient)[0]
    Tensor.realize(output, state_gradient)
    return output.tolist(), state_gradient.tolist()


class GraphTest(unittest.TestCase):
    def test_owns_ordered_immutable_identity(self):
        source, target = SOURCE.copy(), TARGET.copy()
        graph = Graph(6, source, target)
        source[0], target[0] = 0, 0

        self.assertEqual(graph.nodes, 6)
        self.assertEqual(graph.source, tuple(SOURCE))
        self.assertEqual(graph.target, tuple(TARGET))
        self.assertEqual(graph.edges, 6)
        self.assertEqual(graph, Graph(6, SOURCE, TARGET))
        self.assertEqual(hash(graph), hash(Graph(6, SOURCE, TARGET)))
        self.assertNotEqual(graph, Graph(6, list(reversed(SOURCE)), list(reversed(TARGET))))
        with self.assertRaises(FrozenInstanceError):
            setattr(graph, "nodes", 7)

    def test_rejects_invalid_edges(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            Graph(0, [], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            Graph(2, [0], [])
        with self.assertRaisesRegex(ValueError, r"\[0, 2\)"):
            Graph(2, [0], [2])

    def test_in_degree_does_not_expose_mutable_topology(self):
        graph = Graph(6, SOURCE, TARGET)
        expected = [0, 1, 4, 0, 1, 0]
        degree = graph.in_degree(device=Device.DEFAULT).realize()

        self.assertEqual(degree.tolist(), expected)
        degree.assign(Tensor.zeros(6, dtype=degree.dtype, device=Device.DEFAULT)).realize()
        self.assertEqual(graph.in_degree(device=Device.DEFAULT).tolist(), expected)

    def test_cartesian_product_is_left_major_and_sparse(self):
        time = Graph(3, [0, 1], [1, 2])
        space = Graph(2, [0, 1], [1, 0])
        product = time.cartesian(space)

        self.assertEqual(product.nodes, 6)
        self.assertEqual(product.source, (0, 1, 2, 3, 0, 1, 2, 3, 4, 5))
        self.assertEqual(product.target, (2, 3, 4, 5, 1, 0, 3, 2, 5, 4))
        self.assertEqual(product.edges, time.edges * space.nodes + time.nodes * space.edges)


class CSRBackendTest(unittest.TestCase):
    def test_groups_both_directions_without_merging_duplicates(self):
        csr = Graph(6, SOURCE, TARGET)._csr

        self.assertEqual(csr.row_ptr, (0, 0, 1, 5, 5, 6, 6))
        self.assertEqual(csr.column, (0, 0, 1, 1, 3, 1))
        self.assertEqual(csr.transpose_row_ptr, (0, 2, 5, 5, 6, 6, 6))
        self.assertEqual(csr.transpose_column, (1, 2, 2, 2, 4, 2))
        self.assertEqual(csr.edge_order, (1, 5, 0, 2, 3, 4))
        self.assertEqual(csr.transpose_order, (1, 5, 0, 2, 4, 3))

    def test_reuses_realized_connectivity(self):
        csr = Graph(6, SOURCE, TARGET)._csr

        self.assertIs(csr._tensors(Device.DEFAULT), csr._tensors(Device.DEFAULT))

    def test_reuses_realized_edge_maps(self):
        csr = Graph(6, SOURCE, TARGET)._csr
        edge_order, transpose_order, source, target = csr._edge_tensors(Device.DEFAULT)

        self.assertIs(csr._edge_tensors(Device.DEFAULT), csr._edge_tensors(Device.DEFAULT))
        self.assertEqual(edge_order.tolist(), [1, 5, 0, 2, 3, 4])
        self.assertEqual(transpose_order.tolist(), [1, 5, 0, 2, 4, 3])
        self.assertEqual(source.tolist(), SOURCE)
        self.assertEqual(target.tolist(), TARGET)


class GraphSumTest(unittest.TestCase):
    def test_rejects_incompatible_values(self):
        graph = Graph(3, [], [])

        with self.assertRaisesRegex(ValueError, r"shape \[\.\.\., N, H\]"):
            graph.sum(Tensor.ones(3, device=Device.DEFAULT))
        with self.assertRaisesRegex(ValueError, "3 node rows"):
            graph.sum(Tensor.ones(2, 2, 1, device=Device.DEFAULT))

    def test_empty_graph(self):
        graph = Graph(3, [], [])
        state = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=Device.DEFAULT).realize()
        output = graph.sum(state)
        gradient = output.sum().gradient(state)[0]

        self.assertEqual(output.tolist(), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(gradient.tolist(), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def test_single_node(self):
        graph = Graph(1, [0, 0, 0], [0, 0, 0])
        state = Tensor([[2.0, 3.0]], device=Device.DEFAULT).realize()
        output = graph.sum(state)
        gradient = output.gradient(
            state,
            gradient=Tensor([[5.0, 7.0]], device=Device.DEFAULT),
        )[0]

        self.assertEqual(output.tolist(), [[6.0, 9.0]])
        self.assertEqual(gradient.tolist(), [[15.0, 21.0]])

    def test_scalar_feature_empty_rows(self):
        graph = Graph(4, [0, 0], [0, 0])
        state = Tensor.ones(4, 1, device=Device.DEFAULT).realize()
        output = graph.sum(state)
        gradient = output.sum().gradient(state)[0]
        Tensor.realize(output, gradient)

        self.assertEqual(output.tolist(), [[2.0], [0.0], [0.0], [0.0]])
        self.assertEqual(gradient.tolist(), [[2.0], [0.0], [0.0], [0.0]])

    def test_forward_and_gradient(self):
        output, gradient = run()
        self.assertEqual(output, EXPECTED)
        self.assertEqual(gradient, EXPECTED_GRADIENT)

    def test_batch_and_time_axes_share_one_sparse_call(self):
        graph = Graph(6, SOURCE, TARGET)
        lanes = [
            [VALUES, [[2 * value for value in row] for row in VALUES]],
            [list(reversed(VALUES)), VALUES],
        ]
        state = Tensor(lanes, device=Device.DEFAULT).realize()
        output = graph.sum(state)
        gradient = output.sum().gradient(state)[0]

        def aggregate(values):
            result = [[0.0, 0.0] for _ in values]
            for source, target in zip(SOURCE, TARGET):
                for feature, value in enumerate(values[source]):
                    result[target][feature] += value
            return result

        expected = [[aggregate(values) for values in batch] for batch in lanes]
        out_degree = [SOURCE.count(node) for node in range(6)]
        expected_gradient = [
            [[[float(out_degree[node])] * 2 for node in range(6)] for _ in batch]
            for batch in lanes
        ]
        self._assert_sparse_kernel(output, nodes=6, edges=6, width=8)
        self._assert_sparse_kernel(gradient, nodes=6, edges=6, width=8)
        self.assertEqual(output.tolist(), expected)
        self.assertEqual(gradient.tolist(), expected_gradient)

    def test_edge_order_does_not_change_sum(self):
        output, gradient = run(list(reversed(SOURCE)), list(reversed(TARGET)))
        self.assertEqual(output, EXPECTED)
        self.assertEqual(gradient, EXPECTED_GRADIENT)

    def test_canonical_float_accumulation(self):
        values = Tensor([[1e20], [-1e20], [1.0]], device=Device.DEFAULT).realize()
        forward = Graph(3, [0, 1, 2], [0, 0, 0])
        reverse = Graph(3, [2, 1, 0], [0, 0, 0])

        self.assertEqual(reverse.sum(values).tolist(), forward.sum(values).tolist())

    def test_exact_fixture_is_vertex_permutation_equivariant(self):
        old_to_new = [2, 0, 5, 1, 4, 3]
        values = [[0.0, 0.0] for _ in VALUES]
        gradient_values = [[0.0, 0.0] for _ in GRADIENT]
        for old, new in enumerate(old_to_new):
            values[new] = VALUES[old]
            gradient_values[new] = GRADIENT[old]

        output, gradient = run(
            [old_to_new[node] for node in SOURCE],
            [old_to_new[node] for node in TARGET],
            values,
            gradient_values,
        )
        for old, new in enumerate(old_to_new):
            self.assertEqual(output[new], EXPECTED[old])
            self.assertEqual(gradient[new], EXPECTED_GRADIENT[old])

    def test_forward_and_backward_have_sparse_structure(self):
        source = [0, 1, 1, 2, 3, 4, 4]
        target = [1, 0, 3, 3, 3, 0, 3]
        graph = Graph(5, source, target)
        state = Tensor.ones(5, 3, device=Device.DEFAULT).realize()
        output = graph.sum(state)
        gradient = output.gradient(state, gradient=Tensor.ones(5, 3, device=Device.DEFAULT))[0]

        self._assert_sparse_kernel(output, nodes=5, edges=7, width=3)
        self._assert_sparse_kernel(gradient, nodes=5, edges=7, width=3)

    def _assert_sparse_kernel(self, tensor: Tensor, *, nodes: int, edges: int, width: int) -> None:
        calls = [uop for uop in tensor.uop.toposort() if uop.op is Ops.CALL]
        self.assertEqual(len(calls), 1)
        body = calls[0].src[0]
        loop = UOp.loop(-1)
        loops = [
            uop
            for uop in body.toposort()
            if (uop.op, uop.dtype) == (loop.op, loop.dtype)
        ]
        self.assertEqual(len(loops), 1)
        ranges = [
            uop
            for uop in body.toposort()
            if uop.op is Ops.RANGE and uop not in loops
        ]
        self.assertEqual(len(ranges), 1)
        self.assertEqual(int(ranges[0].src[0]), nodes * width)
        limit = max(nodes * width, edges, nodes + 1)
        self.assertTrue(all(
            prod(int(size) for size in uop._shape) <= limit
            for uop in body.toposort()
            if uop._shape is not None
        ))


class GraphMeanTest(unittest.TestCase):
    def test_leading_axes_empty_rows_and_gradient(self):
        graph = Graph(4, [0, 1, 1], [2, 2, 3])
        values = Tensor(
            [
                [[2.0], [4.0], [8.0], [16.0]],
                [[4.0], [8.0], [16.0], [32.0]],
            ],
            device=Device.DEFAULT,
        ).realize()
        output = graph.mean(values)
        gradient = output.sum().gradient(values)[0]

        self.assertEqual(
            output.tolist(),
            [
                [[0.0], [0.0], [3.0], [4.0]],
                [[0.0], [0.0], [6.0], [8.0]],
            ],
        )
        self.assertEqual(
            gradient.tolist(),
            [
                [[0.5], [1.5], [0.0], [0.0]],
                [[0.5], [1.5], [0.0], [0.0]],
            ],
        )

    def test_rejects_integer_values(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            Graph(2, [0], [1]).mean(Tensor.ones(2, 1, dtype=dtypes.int32))


if __name__ == "__main__":
    unittest.main()
