import unittest

from tinygrad import Tensor

from experiments.sparse_aggregation import measure, native_edge_sum


class SparseAggregationFeasibilityTest(unittest.TestCase):
    def test_forward_and_gradient(self) -> None:
        state = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device="CPU")
        source = Tensor([0, 1, 1, 3, 2], device="CPU")
        target = Tensor([1, 2, 0, 2, 1], device="CPU")
        output = native_edge_sum(state, source, target, 4)

        self.assertEqual(output.tolist(), [[3.0, 4.0], [6.0, 8.0], [10.0, 12.0], [0.0, 0.0]])
        gradient = output.square().sum().gradient(state)[0]
        self.assertEqual(
            gradient.tolist(),
            [[12.0, 16.0], [26.0, 32.0], [12.0, 16.0], [20.0, 24.0]],
        )

    def test_gradient_matches_finite_difference(self) -> None:
        values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        source = Tensor([0, 1, 1, 3, 2], device="CPU")
        target = Tensor([1, 2, 0, 2, 1], device="CPU")
        state = Tensor(values, device="CPU")
        analytic = native_edge_sum(state, source, target, 4).square().sum().gradient(state)[0]

        epsilon = 1e-3
        plus = [row.copy() for row in values]
        minus = [row.copy() for row in values]
        plus[1][0] += epsilon
        minus[1][0] -= epsilon
        upper = native_edge_sum(Tensor(plus, device="CPU"), source, target, 4).square().sum().item()
        lower = (
            native_edge_sum(Tensor(minus, device="CPU"), source, target, 4)
            .square()
            .sum()
            .item()
        )

        self.assertAlmostEqual(analytic.tolist()[1][0], (upper - lower) / (2 * epsilon), delta=0.02)

    def test_vertex_permutation_equivariance(self) -> None:
        values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        source = [0, 1, 1, 3, 2]
        target = [1, 2, 0, 2, 1]
        old_to_new = [2, 0, 3, 1]

        expected = native_edge_sum(
            Tensor(values, device="CPU"),
            Tensor(source, device="CPU"),
            Tensor(target, device="CPU"),
            4,
        ).tolist()
        permuted_values = [[0.0, 0.0] for _ in values]
        for old, new in enumerate(old_to_new):
            permuted_values[new] = values[old]
        actual = native_edge_sum(
            Tensor(permuted_values, device="CPU"),
            Tensor([old_to_new[node] for node in source], device="CPU"),
            Tensor([old_to_new[node] for node in target], device="CPU"),
            4,
        ).tolist()

        for old, new in enumerate(old_to_new):
            self.assertEqual(actual[new], expected[old])

    def test_native_candidate_fails_sparse_scaling_gate(self) -> None:
        small = measure(32, 64, 4, device="CPU")
        large = measure(64, 128, 4, device="CPU")

        self.assertGreater(small.dense_carrier_count, 0)
        self.assertGreater(large.dense_carrier_count, 0)
        self.assertGreater(large.operations / small.operations, 3.5)


if __name__ == "__main__":
    unittest.main()
