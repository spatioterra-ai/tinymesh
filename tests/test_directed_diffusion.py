import unittest
from dataclasses import astuple, dataclass
from math import prod

from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import Ops

from tinymesh import Graph
from tinymesh.nn import DirectedDiffusion


SOURCE = [0, 0, 1, 2, 2, 2]
TARGET = [1, 2, 2, 0, 1, 1]
AFFINITY = [1.0, 3.0, 2.0, 4.0, 2.0, 1.0]
VALUES = [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0], [5.0, 2.0], [7.0, -3.0]]
FORWARD_GRADIENT = [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0], [2.0, 0.0], [5.0, -3.0]]
REVERSE_GRADIENT = [[-2.0, 1.0], [1.0, 3.0], [4.0, -1.0], [0.0, 2.0], [-3.0, 5.0]]


@dataclass(frozen=True)
class DiffusionResult:
    forward_weight: list[float]
    reverse_weight: list[float]
    forward: list[list[float]]
    reverse: list[list[float]]
    value_gradient: list[list[float]]
    affinity_gradient: list[float]


def reference(
    source=SOURCE,
    target=TARGET,
    affinity=AFFINITY,
    values=VALUES,
    forward_gradient=FORWARD_GRADIENT,
    reverse_gradient=REVERSE_GRADIENT,
):
    nodes, features = len(values), len(values[0])
    outgoing, incoming = [0.0] * nodes, [0.0] * nodes
    for edge_source, edge_target, edge_affinity in zip(source, target, affinity):
        outgoing[edge_source] += edge_affinity
        incoming[edge_target] += edge_affinity

    forward_weight = [
        edge_affinity / outgoing[edge_source]
        for edge_source, edge_affinity in zip(source, affinity)
    ]
    reverse_weight = [
        edge_affinity / incoming[edge_target]
        for edge_target, edge_affinity in zip(target, affinity)
    ]
    forward = [[0.0] * features for _ in range(nodes)]
    reverse = [[0.0] * features for _ in range(nodes)]
    value_gradient = [[0.0] * features for _ in range(nodes)]
    forward_score, reverse_score = [], []
    for edge_source, edge_target, forward_scale, reverse_scale in zip(
        source, target, forward_weight, reverse_weight
    ):
        forward_score.append(sum(
            values[edge_source][feature] * forward_gradient[edge_target][feature]
            for feature in range(features)
        ))
        reverse_score.append(sum(
            values[edge_target][feature] * reverse_gradient[edge_source][feature]
            for feature in range(features)
        ))
        for feature in range(features):
            forward[edge_target][feature] += forward_scale * values[edge_source][feature]
            reverse[edge_source][feature] += reverse_scale * values[edge_target][feature]
            value_gradient[edge_source][feature] += (
                forward_scale * forward_gradient[edge_target][feature]
            )
            value_gradient[edge_target][feature] += (
                reverse_scale * reverse_gradient[edge_source][feature]
            )

    forward_total, reverse_total = [0.0] * nodes, [0.0] * nodes
    for edge_source, edge_target, edge_affinity, forward_value, reverse_value in zip(
        source, target, affinity, forward_score, reverse_score
    ):
        forward_total[edge_source] += edge_affinity * forward_value
        reverse_total[edge_target] += edge_affinity * reverse_value
    affinity_gradient = [
        (
            (forward_value * outgoing[edge_source] - forward_total[edge_source])
            / outgoing[edge_source] ** 2
            + (reverse_value * incoming[edge_target] - reverse_total[edge_target])
            / incoming[edge_target] ** 2
        )
        for edge_source, edge_target, forward_value, reverse_value in zip(
            source, target, forward_score, reverse_score
        )
    ]
    return DiffusionResult(
        forward_weight,
        reverse_weight,
        forward,
        reverse,
        value_gradient,
        affinity_gradient,
    )


def run(source=SOURCE, target=TARGET, affinity=AFFINITY, device=Device.DEFAULT):
    graph = Graph(len(VALUES), source, target)
    affinity_tensor = Tensor(affinity, device=device).realize()
    values = Tensor(VALUES, device=device).realize()
    diffusion = DirectedDiffusion(graph, affinity_tensor)
    forward, reverse = diffusion(values)
    loss = (
        forward * Tensor(FORWARD_GRADIENT, device=device)
        + reverse * Tensor(REVERSE_GRADIENT, device=device)
    ).sum()
    value_gradient, affinity_gradient = loss.gradient(values, affinity_tensor)
    Tensor.realize(forward, reverse, value_gradient, affinity_gradient)
    return DiffusionResult(
        diffusion.forward_weight.tolist(),
        diffusion.reverse_weight.tolist(),
        forward.tolist(),
        reverse.tolist(),
        value_gradient.tolist(),
        affinity_gradient.tolist(),
    )


class DirectedDiffusionTest(unittest.TestCase):
    def assert_nested_close(self, actual, expected) -> None:
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_close(actual_item, expected_item)
        else:
            self.assertAlmostEqual(actual, expected, places=5)

    def test_values_and_gradients_match_host_reference(self) -> None:
        self.assert_nested_close(astuple(run()), astuple(reference()))

    def test_reverse_graph_keeps_original_edge_order(self) -> None:
        graph = Graph(len(VALUES), SOURCE, TARGET)
        diffusion = DirectedDiffusion(graph, Tensor(AFFINITY, device=Device.DEFAULT))

        self.assertEqual(diffusion.reverse.source, tuple(TARGET))
        self.assertEqual(diffusion.reverse.target, tuple(SOURCE))

    def test_edge_order_moves_weights_and_gradients_with_connectivity(self) -> None:
        actual = run(
            list(reversed(SOURCE)),
            list(reversed(TARGET)),
            list(reversed(AFFINITY)),
        )
        expected = run()

        for name in ("forward_weight", "reverse_weight", "affinity_gradient"):
            self.assert_nested_close(
                getattr(actual, name),
                list(reversed(getattr(expected, name))),
            )
        for name in ("forward", "reverse", "value_gradient"):
            self.assert_nested_close(getattr(actual, name), getattr(expected, name))

    def test_leading_axis_matches_independent_lanes(self) -> None:
        graph = Graph(len(VALUES), SOURCE, TARGET)
        diffusion = DirectedDiffusion(graph, Tensor(AFFINITY, device=Device.DEFAULT))
        values = Tensor(
            [VALUES, [[2 * value for value in row] for row in VALUES]],
            device=Device.DEFAULT,
        )

        actual = diffusion(values)
        expected = tuple(
            Tensor.stack(*(diffusion(lane)[direction] for lane in values))
            for direction in range(2)
        )

        self.assert_nested_close(actual[0].tolist(), expected[0].tolist())
        self.assert_nested_close(actual[1].tolist(), expected[1].tolist())

    def test_residual_is_root_relative_and_direction_ordered(self) -> None:
        graph = Graph(len(VALUES), SOURCE, TARGET)
        diffusion = DirectedDiffusion(graph, Tensor(AFFINITY, device=Device.DEFAULT))
        root, expected = Tensor(VALUES), reference()
        residual = (Tensor(expected.forward) - root).cat(Tensor(expected.reverse) - root, dim=-1)

        self.assert_nested_close(diffusion.residual(root).tolist(), residual.tolist())

    def test_empty_graph_returns_zero(self) -> None:
        graph = Graph(3, [], [])
        affinity = Tensor([], device=Device.DEFAULT).realize()
        values = Tensor([[1.0], [2.0], [3.0]], device=Device.DEFAULT).realize()
        diffusion = DirectedDiffusion(graph, affinity)
        forward, reverse = diffusion(values)
        value_gradient, affinity_gradient = (forward + reverse).sum().gradient(
            values,
            affinity,
        )

        self.assertEqual(forward.tolist(), [[0.0], [0.0], [0.0]])
        self.assertEqual(reverse.tolist(), [[0.0], [0.0], [0.0]])
        self.assertEqual(value_gradient.tolist(), [[0.0], [0.0], [0.0]])
        self.assertEqual(affinity_gradient.tolist(), [])

    def test_rejects_incompatible_affinity(self) -> None:
        graph = Graph(2, [0], [1])
        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            DirectedDiffusion(graph, Tensor.ones(1, 1))
        with self.assertRaisesRegex(ValueError, r"shape \[1\]"):
            DirectedDiffusion(graph, Tensor.ones(2))
        with self.assertRaisesRegex(ValueError, "floating"):
            DirectedDiffusion(graph, Tensor.ones(1, dtype=dtypes.int32))

    def test_application_stays_sparse(self) -> None:
        nodes, edges, features = 5, len(SOURCE), 3
        graph = Graph(nodes, SOURCE, TARGET)
        diffusion = DirectedDiffusion(
            graph,
            Tensor(AFFINITY, device=Device.DEFAULT).realize(),
        )
        Tensor.realize(diffusion.forward_weight, diffusion.reverse_weight)
        values = Tensor.ones(nodes, features, device=Device.DEFAULT).realize()
        forward, reverse = diffusion(values)
        output = diffusion.residual(values)

        calls = [
            uop for uop in output.uop.toposort()
            if uop.op is Ops.CALL and uop.src[0].arg.name == "csr_sum"
        ]
        self.assertEqual(len(calls), 2)
        self.assertEqual(diffusion.forward_weight.shape, (edges,))
        self.assertEqual(diffusion.reverse_weight.shape, (edges,))
        forbidden = {(nodes, nodes), (nodes, edges)}
        for tensor in (forward, reverse):
            shapes = {
                tuple(int(size) for size in uop._shape)
                for uop in tensor.uop.toposort()
                if uop._shape is not None
            }
            self.assertTrue(forbidden.isdisjoint(shapes))
            self.assertLessEqual(
                max(prod(shape) for shape in shapes),
                max(nodes * features, edges, nodes + 1),
            )


if __name__ == "__main__":
    unittest.main()
