import unittest
from math import exp, sqrt, tanh

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from experiments.tgcn import TGCN, fit_one_step
from tinymesh import Graph


SOURCE = [0, 1, 2, 0, 1, 1, 2]
TARGET = [0, 1, 2, 1, 0, 2, 1]
SNAPSHOTS = [[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]]
PARAMETERS = {
    "update": (0.5, 0.7, -0.2, 0.1),
    "reset": (-0.25, 0.4, 0.3, -0.1),
    "candidate": (0.8, 1.1, 0.5, 0.05),
}


def sigmoid(value):
    return 1 / (1 + exp(-value))


def graph_convolution(values, weight):
    degree = [0] * 3
    for target in TARGET:
        degree[target] += 1
    scale = [1 / sqrt(value) for value in degree]
    output = [0.0] * 3
    for source, target in zip(SOURCE, TARGET):
        output[target] += values[source] * weight * scale[source] * scale[target]
    return output


def reference(snapshots=SNAPSHOTS):
    hidden = [0.0] * 3
    for values in snapshots:
        graph_update = graph_convolution(values, PARAMETERS["update"][0])
        graph_reset = graph_convolution(values, PARAMETERS["reset"][0])
        graph_candidate = graph_convolution(values, PARAMETERS["candidate"][0])
        update = [
            sigmoid(PARAMETERS["update"][1] * value + PARAMETERS["update"][2] * state + PARAMETERS["update"][3])
            for value, state in zip(graph_update, hidden)
        ]
        reset = [
            sigmoid(PARAMETERS["reset"][1] * value + PARAMETERS["reset"][2] * state + PARAMETERS["reset"][3])
            for value, state in zip(graph_reset, hidden)
        ]
        candidate = [
            tanh(
                PARAMETERS["candidate"][1] * value
                + PARAMETERS["candidate"][2] * state * gate
                + PARAMETERS["candidate"][3]
            )
            for value, state, gate in zip(graph_candidate, hidden, reset)
        ]
        hidden = [
            gate * state + (1 - gate) * proposal
            for gate, state, proposal in zip(update, hidden, candidate)
        ]
    return hidden


def run(snapshots=SNAPSHOTS):
    graph = Graph(3, SOURCE, TARGET)
    model = TGCN(1, 1)
    model.graph_projection.linear.weight = Tensor(
        [[PARAMETERS[name][0]] for name in ("update", "reset", "candidate")],
        device=Device.DEFAULT,
    ).realize()
    for name, (_, input_weight, hidden_weight, bias) in PARAMETERS.items():
        gate = getattr(model, name)
        gate.weight = Tensor([[input_weight, hidden_weight]], device=Device.DEFAULT).realize()
        gate.bias = Tensor([bias], device=Device.DEFAULT).realize()

    first, *rest = snapshots
    hidden = model(
        Tensor([[value] for value in first], device=Device.DEFAULT).realize(),
        graph,
    )
    for snapshot in rest:
        values = Tensor([[value] for value in snapshot], device=Device.DEFAULT).realize()
        hidden = model(values, graph, hidden)
    return hidden


class TGCNTest(unittest.TestCase):
    def test_matches_reference_across_snapshots(self):
        actual = run().flatten().tolist()
        for actual_value, expected_value in zip(actual, reference()):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_snapshot_order_changes_final_state(self):
        forward = run().flatten().tolist()
        reverse = run(list(reversed(SNAPSHOTS))).flatten().tolist()
        self.assertTrue(any(abs(left - right) > 1e-3 for left, right in zip(forward, reverse)))

    def test_optimizer_reaches_graph_parameter_through_time(self):
        observation = fit_one_step(Device.DEFAULT)
        self.assertAlmostEqual(observation.initial_loss, 0.718740, places=5)
        self.assertAlmostEqual(observation.candidate_gradient, -0.188622, places=5)
        self.assertAlmostEqual(observation.final_loss, 0.686385, places=5)
        self.assertAlmostEqual(observation.candidate_weight, 1.188622, places=5)

    def test_gate_graph_projections_share_one_sparse_call(self):
        model = TGCN(1, 1)
        graph = Graph(3, SOURCE, TARGET)
        output = model(Tensor.zeros(3, 1, device=Device.DEFAULT), graph)
        calls = [uop for uop in output.uop.toposort() if uop.op is Ops.CALL]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].src[0].arg.name, "csr_sum")

    def test_rejects_incompatible_hidden_state(self):
        model = TGCN(1, 2)
        graph = Graph(3, SOURCE, TARGET)
        values = Tensor.zeros(3, 1, device=Device.DEFAULT)
        with self.assertRaisesRegex(ValueError, r"shape \(3, 2\)"):
            model(values, graph, Tensor.zeros(3, 1, device=Device.DEFAULT))


if __name__ == "__main__":
    unittest.main()
