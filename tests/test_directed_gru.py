import unittest
from math import exp, tanh

from tinygrad import Context, Device, Tensor, nn
from tinygrad.uop.ops import Ops

from experiments.directed_diffusion import DirectedDiffusion
from experiments.directed_gru import DiffusionForecast, DiffusionGRU
from tinymesh import Graph


SOURCE = [0, 0, 1, 2]
TARGET = [1, 2, 2, 0]
AFFINITY = [1.0, 3.0, 2.0, 4.0]
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


def diffuse(values):
    outgoing, incoming = [0.0] * 3, [0.0] * 3
    for source, target, affinity in zip(SOURCE, TARGET, AFFINITY):
        outgoing[source] += affinity
        incoming[target] += affinity
    forward, reverse = [[0.0] * len(values[0]) for _ in range(3)], [
        [0.0] * len(values[0]) for _ in range(3)
    ]
    for source, target, affinity in zip(SOURCE, TARGET, AFFINITY):
        for feature in range(len(values[0])):
            forward[target][feature] += affinity / outgoing[source] * values[source][feature]
            reverse[source][feature] += affinity / incoming[target] * values[target][feature]
    return forward, reverse


def linear(values, weight, bias):
    return [
        [
            bias[output] + sum(
                coefficient * value
                for coefficient, value in zip(weight[output], row)
            )
            for output in range(len(weight))
        ]
        for row in values
    ]


def basis(values):
    forward, reverse = diffuse(values)
    return [
        row + forward_row + reverse_row
        for row, forward_row, reverse_row in zip(values, forward, reverse)
    ]


def reference():
    hidden = [[0.0] for _ in range(3)]
    for snapshot in SNAPSHOTS:
        values = [[value] for value in snapshot]
        gates = linear(
            basis([value + state for value, state in zip(values, hidden)]),
            GATE_WEIGHT,
            GATE_BIAS,
        )
        update = [sigmoid(row[0]) for row in gates]
        reset = [sigmoid(row[1]) for row in gates]
        candidate = [
            row[0] for row in linear(
                basis([
                    value + [state[0] * gate]
                    for value, state, gate in zip(values, hidden, reset)
                ]),
                CANDIDATE_WEIGHT,
                CANDIDATE_BIAS,
            )
        ]
        hidden = [
            [gate * state[0] + (1 - gate) * tanh(proposal)]
            for gate, state, proposal in zip(update, hidden, candidate)
        ]
    return hidden


def diffusion(nodes: int = 3) -> DirectedDiffusion:
    operator = DirectedDiffusion(
        Graph(nodes, SOURCE, TARGET),
        Tensor(AFFINITY, device=Device.DEFAULT).realize(),
    )
    Tensor.realize(operator.forward_weight, operator.reverse_weight)
    return operator


def model() -> DiffusionGRU:
    cell = DiffusionGRU(1, 1)
    cell.gates.weight = Tensor(GATE_WEIGHT, device=Device.DEFAULT).realize()
    cell.gates.bias = Tensor(GATE_BIAS, device=Device.DEFAULT).realize()
    cell.candidate.weight = Tensor(CANDIDATE_WEIGHT, device=Device.DEFAULT).realize()
    cell.candidate.bias = Tensor(CANDIDATE_BIAS, device=Device.DEFAULT).realize()
    return cell


class DiffusionGRUTest(unittest.TestCase):
    def test_matches_host_reference_across_time(self) -> None:
        cell, operator = model(), diffusion()
        hidden = None
        for snapshot in SNAPSHOTS:
            hidden = cell(
                Tensor([[value] for value in snapshot], device=Device.DEFAULT),
                operator,
                hidden,
            )

        for actual_row, expected_row in zip(hidden.tolist(), reference()):
            self.assertAlmostEqual(actual_row[0], expected_row[0], places=5)

    def test_batch_axis_matches_independent_hidden_states(self) -> None:
        cell, operator = DiffusionGRU(1, 2), diffusion()
        values = Tensor(
            [
                [[1.0], [0.0], [-1.0]],
                [[0.5], [-0.5], [1.0]],
            ],
            device=Device.DEFAULT,
        ).realize()

        expected = Tensor.stack(*(cell(lane, operator) for lane in values))
        actual = cell(values, operator)
        for actual_batch, expected_batch in zip(actual.tolist(), expected.tolist()):
            for actual_row, expected_row in zip(actual_batch, expected_batch):
                for actual_value, expected_value in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_realized_inference_matches_lazy_unroll(self) -> None:
        Tensor.manual_seed(0)
        forecast, operator = DiffusionForecast(1, 2), diffusion()
        values = Tensor(
            [[[[1.0], [0.0], [-1.0]], [[0.5], [-0.5], [1.0]]]],
            device=Device.DEFAULT,
        ).realize()

        lazy = forecast(values, operator)
        stepped = forecast(values, operator, realize_steps=True)

        for actual, expected in zip(stepped.flatten().tolist(), lazy.flatten().tolist()):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_one_step_has_four_sparse_calls(self) -> None:
        nodes = 7
        output = DiffusionGRU(1, 1)(
            Tensor.zeros(nodes, 1, device=Device.DEFAULT),
            diffusion(nodes),
        )
        calls = [
            uop for uop in output.uop.toposort()
            if uop.op is Ops.CALL and uop.src[0].arg.name == "csr_sum"
        ]
        self.assertEqual(len(calls), 4)
        shapes = {
            tuple(int(size) for size in uop._shape)
            for uop in output.uop.toposort()
            if uop._shape is not None
        }
        self.assertTrue({(nodes, nodes), (nodes, len(SOURCE))}.isdisjoint(shapes))

    def test_optimizer_reaches_candidate_through_time(self) -> None:
        Tensor.manual_seed(0)
        forecast, operator = DiffusionForecast(1, 2), diffusion()
        values = Tensor(
            [[[[1.0], [0.0], [-1.0]], [[0.5], [-0.5], [1.0]]]],
            device=Device.DEFAULT,
        ).realize()
        target = Tensor.zeros(1, 3, 1, device=Device.DEFAULT).realize()
        optimizer = nn.optim.SGD(nn.state.get_parameters(forecast), lr=0.01)

        initial = (forecast(values, operator) - target).square().mean()
        with Context(TRAINING=1):
            optimizer.zero_grad()
            loss = (forecast(values, operator) - target).square().mean().backward()
            gradient = forecast.cell.candidate.weight.grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertNotEqual(gradient.abs().sum().item(), 0.0)
            loss.realize(*optimizer.schedule_step())
        self.assertLess(
            (forecast(values, operator) - target).square().mean().item(),
            initial.item(),
        )

    def test_same_seed_gives_identical_diffusion_models(self) -> None:
        states = []
        for _ in range(3):
            Tensor.manual_seed(7)
            states.append({
                name: value.tolist()
                for name, value in nn.state.get_state_dict(
                    DiffusionForecast(1, 2)
                ).items()
            })
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])

    def test_rejects_incompatible_values_or_hidden_state(self) -> None:
        cell, operator = DiffusionGRU(1, 2), diffusion()
        with self.assertRaisesRegex(ValueError, r"shape \[\.\.\., 3, 1\]"):
            cell(Tensor.zeros(2, 1), operator)
        with self.assertRaisesRegex(ValueError, r"shape \(3, 2\)"):
            cell(
                Tensor.zeros(3, 1),
                operator,
                Tensor.zeros(3, 1),
            )


if __name__ == "__main__":
    unittest.main()
