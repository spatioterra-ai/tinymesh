import unittest
from math import prod

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from experiments.csr_aggregation import CSRTopology, csr_edge_sum


SOURCE = [1, 0, 1, 3, 1, 0]
TARGET = [2, 1, 2, 2, 4, 2]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0], [19.0, 23.0], [29.0, 31.0]]
GRADIENT = [[1.0, 10.0], [2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0], [23.0, 29.0]]
EXPECTED = [[0.0, 0.0], [1.0, 2.0], [20.0, 29.0], [0.0, 0.0], [3.0, 5.0], [0.0, 0.0]]
EXPECTED_GRADIENT = [[7.0, 10.0], [27.0, 33.0], [0.0, 0.0], [5.0, 7.0], [0.0, 0.0], [0.0, 0.0]]


def run(source=SOURCE, target=TARGET, values=VALUES, gradient_values=GRADIENT):
    topology = CSRTopology(6, source, target)
    state = Tensor(values, device=Device.DEFAULT).realize()
    output = csr_edge_sum(state, topology)
    gradient = Tensor(gradient_values, device=Device.DEFAULT).realize()
    state_gradient = output.gradient(state, gradient=gradient)[0]
    Tensor.realize(output, state_gradient)
    return output.tolist(), state_gradient.tolist()


class CSRTopologyTest(unittest.TestCase):
    def test_groups_both_directions_without_merging_duplicates(self):
        topology = CSRTopology(6, SOURCE, TARGET)

        self.assertEqual(topology.row_ptr, (0, 0, 1, 5, 5, 6, 6))
        self.assertEqual(topology.column, (0, 0, 1, 1, 3, 1))
        self.assertEqual(topology.transpose_row_ptr, (0, 2, 5, 5, 6, 6, 6))
        self.assertEqual(topology.transpose_column, (1, 2, 2, 2, 4, 2))

    def test_rejects_invalid_edges(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            CSRTopology(0, [], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            CSRTopology(2, [0], [])
        with self.assertRaisesRegex(ValueError, r"\[0, 2\)"):
            CSRTopology(2, [0], [2])
        with self.assertRaises(TypeError):
            CSRTopology(2, (0, 0, 1), (0,), (0, 0, 1), (0,))


class CSRAggregationTest(unittest.TestCase):
    def test_rejects_wrong_node_count(self):
        topology = CSRTopology(3, [], [])
        with self.assertRaisesRegex(ValueError, "3 rows"):
            csr_edge_sum(Tensor.ones(2, 1, device=Device.DEFAULT), topology)

    def test_empty_graph(self):
        topology = CSRTopology(3, [], [])
        state = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=Device.DEFAULT).realize()
        output = csr_edge_sum(state, topology)
        gradient = output.sum().gradient(state)[0]

        self.assertEqual(output.tolist(), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(gradient.tolist(), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def test_single_node(self):
        topology = CSRTopology(1, [0, 0, 0], [0, 0, 0])
        state = Tensor([[2.0, 3.0]], device=Device.DEFAULT).realize()
        output = csr_edge_sum(state, topology)
        gradient = output.gradient(
            state,
            gradient=Tensor([[5.0, 7.0]], device=Device.DEFAULT),
        )[0]

        self.assertEqual(output.tolist(), [[6.0, 9.0]])
        self.assertEqual(gradient.tolist(), [[15.0, 21.0]])

    def test_scalar_feature_empty_rows(self):
        topology = CSRTopology(4, [0, 0], [0, 0])
        state = Tensor.ones(4, 1, device=Device.DEFAULT).realize()
        output = csr_edge_sum(state, topology)
        gradient = output.sum().gradient(state)[0]
        Tensor.realize(output, gradient)

        self.assertEqual(output.tolist(), [[2.0], [0.0], [0.0], [0.0]])
        self.assertEqual(gradient.tolist(), [[2.0], [0.0], [0.0], [0.0]])

    def test_forward_and_gradient(self):
        output, gradient = run()
        self.assertEqual(output, EXPECTED)
        self.assertEqual(gradient, EXPECTED_GRADIENT)

    def test_edge_order_does_not_change_result(self):
        output, gradient = run(list(reversed(SOURCE)), list(reversed(TARGET)))
        self.assertEqual(output, EXPECTED)
        self.assertEqual(gradient, EXPECTED_GRADIENT)

    def test_edge_order_is_canonical_for_float_accumulation(self):
        values = Tensor([[1e20], [-1e20], [1.0]], device=Device.DEFAULT).realize()
        forward = CSRTopology(3, [0, 1, 2], [0, 0, 0])
        reverse = CSRTopology(3, [2, 1, 0], [0, 0, 0])

        expected = csr_edge_sum(values, forward).tolist()
        actual = csr_edge_sum(values, reverse).tolist()

        self.assertEqual(actual, expected)

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
        topology = CSRTopology(5, source, target)
        state = Tensor.ones(5, 3, device=Device.DEFAULT).realize()
        output = csr_edge_sum(state, topology)
        gradient = output.gradient(state, gradient=Tensor.ones(5, 3, device=Device.DEFAULT))[0]

        self._assert_sparse_kernel(output, nodes=5, edges=7, width=3)
        self._assert_sparse_kernel(gradient, nodes=5, edges=7, width=3)

    def _assert_sparse_kernel(self, tensor: Tensor, *, nodes: int, edges: int, width: int) -> None:
        calls = [uop for uop in tensor.uop.toposort() if uop.op is Ops.CALL]
        self.assertEqual(len(calls), 1)
        body = calls[0].src[0]
        self.assertEqual(sum(uop.op is Ops.LOOP for uop in body.toposort()), 1)
        ranges = [uop for uop in body.toposort() if uop.op is Ops.RANGE]
        self.assertEqual(len(ranges), 1)
        self.assertEqual(int(ranges[0].src[0]), nodes * width)
        limit = max(nodes * width, edges, nodes + 1)
        self.assertTrue(all(
            prod(int(size) for size in uop._shape) <= limit
            for uop in body.toposort()
            if uop._shape is not None
        ))


if __name__ == "__main__":
    unittest.main()
