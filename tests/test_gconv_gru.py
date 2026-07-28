import unittest
from math import exp, sqrt, tanh

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from experiments.gconv_gru import ChebConv, GConvGRU, compare
from tinymesh import Graph


SOURCE = [0, 1, 1, 2, 2, 0]
TARGET = [1, 0, 2, 1, 0, 2]
SNAPSHOTS = [[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]]
GATE_WEIGHT = [
    [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
    [-0.3, 0.5, 0.2, -0.4, 0.1, 0.6],
]
GATE_BIAS = [0.1, -0.2]
CANDIDATE_WEIGHT = [[0.4, -0.3, 0.2, 0.5, -0.6, 0.1]]
CANDIDATE_BIAS = [0.05]


def sigmoid(value):
    return 1 / (1 + exp(-value))


def shift(values):
    degree = [0] * len(values)
    for target in TARGET:
        degree[target] += 1
    output = [[0.0] * len(values[0]) for _ in values]
    for source, target in zip(SOURCE, TARGET):
        scale = 1 / sqrt(degree[source] * degree[target])
        for feature, value in enumerate(values[source]):
            output[target][feature] -= value * scale
    return output


def chebyshev(values, weight, bias):
    states = [values, shift(values)]
    states.append([
        [2 * shifted - original for shifted, original in zip(shifted_row, original_row)]
        for shifted_row, original_row in zip(shift(states[-1]), states[-2])
    ])
    result = []
    for node in range(len(values)):
        basis = [value for state in states for value in state[node]]
        result.append([
            bias[feature] + sum(
                coefficient * value
                for coefficient, value in zip(weight[feature], basis)
            )
            for feature in range(len(weight))
        ])
    return result


def reference():
    hidden = [[0.0] for _ in SNAPSHOTS[0]]
    for snapshot in SNAPSHOTS:
        values = [[value] for value in snapshot]
        gates = chebyshev(
            [value + state for value, state in zip(values, hidden)],
            GATE_WEIGHT,
            GATE_BIAS,
        )
        update = [sigmoid(gate[0]) for gate in gates]
        reset = [sigmoid(gate[1]) for gate in gates]
        candidate = chebyshev(
            [
                value + [state[0] * gate]
                for value, state, gate in zip(values, hidden, reset)
            ],
            CANDIDATE_WEIGHT,
            CANDIDATE_BIAS,
        )
        hidden = [
            [gate * state[0] + (1 - gate) * tanh(proposal[0])]
            for gate, state, proposal in zip(update, hidden, candidate)
        ]
    return hidden


def run():
    graph = Graph(3, SOURCE, TARGET)
    model = GConvGRU(1, 1, 3)
    model.gates.linear.weight = Tensor(GATE_WEIGHT, device=Device.DEFAULT).realize()
    model.gates.linear.bias = Tensor(GATE_BIAS, device=Device.DEFAULT).realize()
    model.candidate.linear.weight = Tensor(CANDIDATE_WEIGHT, device=Device.DEFAULT).realize()
    model.candidate.linear.bias = Tensor(CANDIDATE_BIAS, device=Device.DEFAULT).realize()
    hidden = None
    for snapshot in SNAPSHOTS:
        hidden = model(
            Tensor([[value] for value in snapshot], device=Device.DEFAULT).realize(),
            graph,
            hidden,
        )
    return hidden


class ChebConvTest(unittest.TestCase):
    def test_matches_reference(self):
        values = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]
        weight = [[0.2, -0.1, 0.3, 0.4, -0.2, 0.1]]
        layer = ChebConv(2, 1, 3)
        layer.linear.weight = Tensor(weight, device=Device.DEFAULT).realize()
        layer.linear.bias = Tensor([0.05], device=Device.DEFAULT).realize()
        actual = layer(
            Tensor(values, device=Device.DEFAULT).realize(),
            Graph(3, SOURCE, TARGET),
        ).tolist()

        for actual_row, expected_row in zip(actual, chebyshev(values, weight, [0.05])):
            self.assertAlmostEqual(actual_row[0], expected_row[0], places=5)

    def test_order_two_uses_one_sparse_shift(self):
        layer = ChebConv(1, 1, 2)
        layer.linear.weight = Tensor([[0.0, 1.0]], device=Device.DEFAULT).realize()
        layer.linear.bias = Tensor.zeros(1, device=Device.DEFAULT).realize()
        output = layer(
            Tensor([[1.0], [0.0], [0.0]], device=Device.DEFAULT),
            Graph(3, SOURCE, TARGET),
        )

        self.assertEqual(
            sum(uop.src[0].arg.name == "csr_sum" for uop in output.uop.toposort() if uop.op is Ops.CALL),
            1,
        )
        self.assertNotEqual(output.tolist(), [[0.0], [0.0], [0.0]])


class GConvGRUTest(unittest.TestCase):
    def test_matches_reference_across_snapshots(self):
        actual = run().tolist()
        for actual_row, expected_row in zip(actual, reference()):
            self.assertAlmostEqual(actual_row[0], expected_row[0], places=5)

    def test_two_fused_chebyshev_calls_per_step(self):
        model = GConvGRU(1, 1, 2)
        output = model(
            Tensor.zeros(3, 1, device=Device.DEFAULT),
            Graph(3, SOURCE, TARGET),
        )
        calls = [
            uop for uop in output.uop.toposort()
            if uop.op is Ops.CALL and uop.src[0].arg.name == "csr_sum"
        ]
        self.assertEqual(len(calls), 2)

    def test_controlled_comparison_reports_unequal_cost(self):
        observation = compare(Device.DEFAULT)
        self.assertEqual(observation.steps, 1)
        self.assertEqual((observation.tgcn_parameters, observation.gconv_gru_parameters), (12, 15))
        self.assertEqual((observation.tgcn_sparse_calls, observation.gconv_gru_sparse_calls), (1, 2))
        self.assertAlmostEqual(observation.tgcn_initial_loss, observation.gconv_gru_initial_loss, places=6)
        self.assertNotEqual(observation.gconv_gru_spatial_gradient, 0.0)
        self.assertAlmostEqual(
            observation.tgcn_spatial_gradient,
            -observation.gconv_gru_spatial_gradient,
            places=6,
        )
        self.assertLess(observation.tgcn_final_loss, observation.tgcn_initial_loss)
        self.assertLess(observation.gconv_gru_final_loss, observation.gconv_gru_initial_loss)

    def test_rejects_incompatible_hidden_state(self):
        model = GConvGRU(1, 2, 2)
        graph = Graph(3, SOURCE, TARGET)
        with self.assertRaisesRegex(ValueError, r"shape \(3, 2\)"):
            model(
                Tensor.zeros(3, 1, device=Device.DEFAULT),
                graph,
                Tensor.zeros(3, 1, device=Device.DEFAULT),
            )

    def test_rejects_non_symmetric_or_looped_graph(self):
        model = GConvGRU(1, 1, 2)
        values = Tensor.zeros(2, 1, device=Device.DEFAULT)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            model(values, Graph(2, [0], [1]))
        with self.assertRaisesRegex(ValueError, "self-loops"):
            model(values, Graph(2, [0, 1, 0], [0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
