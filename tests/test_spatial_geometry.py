import math
import unittest
from dataclasses import astuple, dataclass
from math import prod

from tinygrad import Device, Tensor, UOp
from tinygrad.uop.ops import AxisType, Ops

from experiments.spatial_geometry import radial_message
from tinymesh import Graph


SOURCE = [0, 1, 0, 2]
TARGET = [2, 2, 1, 3]
POSITION = [[0.0, 0.0], [0.0, 3.0], [4.0, 0.0], [4.0, 3.0]]
VALUES = [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0], [5.0, 2.0]]
OUTPUT_GRADIENT = [[0.0, 0.0], [1.0, -2.0], [3.0, 1.0], [-1.0, 2.0]]
DECAY = 0.25


@dataclass(frozen=True)
class GeometryResult:
    displacement: list[list[float]]
    distance: list[float]
    weight: list[float]
    output: list[list[float]]
    position_gradient: list[list[float]]
    value_gradient: list[list[float]]
    weight_gradient: list[float]
    decay_gradient: float


def reference(
    source=SOURCE,
    target=TARGET,
    position=POSITION,
    values=VALUES,
    output_gradient=OUTPUT_GRADIENT,
    decay=DECAY,
):
    nodes, features = len(position), len(values[0])
    displacement = []
    distance = []
    weight = []
    output = [[0.0] * features for _ in range(nodes)]
    position_gradient = [[0.0] * len(position[0]) for _ in range(nodes)]
    value_gradient = [[0.0] * features for _ in range(nodes)]
    weight_gradient = []
    decay_gradient = 0.0

    for edge_source, edge_target in zip(source, target):
        delta = [
            position[edge_target][axis] - position[edge_source][axis]
            for axis in range(len(position[0]))
        ]
        radius = math.sqrt(sum(value * value for value in delta))
        edge_weight = math.exp(-decay * radius)
        sensitivity = sum(
            values[edge_source][feature] * output_gradient[edge_target][feature]
            for feature in range(features)
        )
        radial_gradient = -decay * edge_weight * sensitivity

        displacement.append(delta)
        distance.append(radius)
        weight.append(edge_weight)
        weight_gradient.append(sensitivity)
        decay_gradient -= radius * edge_weight * sensitivity
        for feature in range(features):
            output[edge_target][feature] += edge_weight * values[edge_source][feature]
            value_gradient[edge_source][feature] += (
                edge_weight * output_gradient[edge_target][feature]
            )
        for axis, delta_value in enumerate(delta):
            delta_gradient = radial_gradient * delta_value / radius
            position_gradient[edge_source][axis] -= delta_gradient
            position_gradient[edge_target][axis] += delta_gradient

    return GeometryResult(
        displacement,
        distance,
        weight,
        output,
        position_gradient,
        value_gradient,
        weight_gradient,
        decay_gradient,
    )


def run(
    source=SOURCE,
    target=TARGET,
    position=POSITION,
    values=VALUES,
    output_gradient=OUTPUT_GRADIENT,
):
    graph = Graph(len(position), source, target)
    position_tensor = Tensor(position, device=Device.DEFAULT).realize()
    values_tensor = Tensor(values, device=Device.DEFAULT).realize()
    decay = Tensor(DECAY, device=Device.DEFAULT).realize()
    displacement, distance, weight, output = radial_message(
        graph,
        position_tensor,
        values_tensor,
        decay,
    )
    gradients = output.gradient(
        position_tensor,
        values_tensor,
        weight,
        decay,
        gradient=Tensor(output_gradient, device=Device.DEFAULT),
    )
    Tensor.realize(displacement, distance, weight, output, *gradients)
    position_gradient, value_gradient, weight_gradient, decay_gradient = gradients
    return GeometryResult(
        displacement.tolist(),
        distance.tolist(),
        weight.tolist(),
        output.tolist(),
        position_gradient.tolist(),
        value_gradient.tolist(),
        weight_gradient.tolist(),
        decay_gradient.item(),
    )


class SpatialGeometryTest(unittest.TestCase):
    def assert_nested_close(self, actual, expected) -> None:
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_close(actual_item, expected_item)
        else:
            self.assertAlmostEqual(actual, expected, places=5)

    def test_values_and_gradients_match_host_reference(self):
        self.assert_nested_close(astuple(run()), astuple(reference()))

    def test_translation_preserves_geometry_and_output(self):
        translated = [[x + 17.0, y - 9.0] for x, y in POSITION]
        actual, expected = run(position=translated), run()

        for name in ("displacement", "distance", "weight", "output"):
            self.assert_nested_close(getattr(actual, name), getattr(expected, name))

    def test_rotation_rotates_displacement_and_preserves_distance(self):
        rotated = [[-y, x] for x, y in POSITION]
        actual = run(position=rotated)
        expected = run()

        self.assert_nested_close(actual.displacement, [
            [-delta_y, delta_x]
            for delta_x, delta_y in expected.displacement
        ])
        for name in ("distance", "weight", "output"):
            self.assert_nested_close(getattr(actual, name), getattr(expected, name))

    def test_edge_order_and_duplicate_edges_preserve_output_and_gradients(self):
        source = [0, 1, 0, 0]
        target = [2, 2, 1, 2]
        forward = run(source=source, target=target)
        reverse = run(source=list(reversed(source)), target=list(reversed(target)))

        for name in ("displacement", "distance", "weight", "weight_gradient"):
            self.assert_nested_close(
                getattr(reverse, name),
                list(reversed(getattr(forward, name))),
            )
        for name in ("output", "position_gradient", "value_gradient", "decay_gradient"):
            self.assert_nested_close(getattr(reverse, name), getattr(forward, name))

    def test_vertex_relabeling_relabels_output(self):
        old_to_new = [2, 0, 3, 1]
        position = [[0.0, 0.0] for _ in POSITION]
        values = [[0.0, 0.0] for _ in VALUES]
        output_gradient = [[0.0, 0.0] for _ in OUTPUT_GRADIENT]
        for old, new in enumerate(old_to_new):
            position[new] = POSITION[old]
            values[new] = VALUES[old]
            output_gradient[new] = OUTPUT_GRADIENT[old]

        original = run()
        relabeled = run(
            source=[old_to_new[node] for node in SOURCE],
            target=[old_to_new[node] for node in TARGET],
            position=position,
            values=values,
            output_gradient=output_gradient,
        )
        for name in ("output", "position_gradient", "value_gradient"):
            original_values = getattr(original, name)
            relabeled_values = getattr(relabeled, name)
            for old, new in enumerate(old_to_new):
                self.assert_nested_close(relabeled_values[new], original_values[old])
        self.assert_nested_close(relabeled.weight_gradient, original.weight_gradient)
        self.assert_nested_close(relabeled.decay_gradient, original.decay_gradient)

    def test_empty_graph(self):
        graph = Graph(3, [], [])
        position = Tensor(POSITION[:3], device=Device.DEFAULT).realize()
        values = Tensor(VALUES[:3], device=Device.DEFAULT).realize()
        decay = Tensor(DECAY, device=Device.DEFAULT).realize()
        displacement, distance, weight, output = radial_message(
            graph,
            position,
            values,
            decay,
        )
        position_gradient, value_gradient, decay_gradient = output.sum().gradient(
            position,
            values,
            decay,
        )

        self.assertEqual(displacement.shape, (0, 2))
        self.assertEqual(distance.shape, (0,))
        self.assertEqual(weight.shape, (0,))
        self.assertEqual(output.tolist(), [[0.0, 0.0]] * 3)
        self.assertEqual(position_gradient.tolist(), [[0.0, 0.0]] * 3)
        self.assertEqual(value_gradient.tolist(), [[0.0, 0.0]] * 3)
        self.assertEqual(decay_gradient.item(), 0.0)

    def test_intermediate_shapes_stay_edge_linear(self):
        nodes, edges, dimensions, features = 5, 7, 3, 2
        graph = Graph(
            nodes,
            source=[0, 1, 1, 2, 3, 4, 4],
            target=[1, 0, 3, 3, 3, 0, 3],
        )
        position = Tensor.ones(nodes, dimensions, device=Device.DEFAULT).realize()
        values = Tensor.ones(nodes, features, device=Device.DEFAULT).realize()
        decay = Tensor(DECAY, device=Device.DEFAULT).realize()
        displacement, distance, weight, output = radial_message(
            graph,
            position,
            values,
            decay,
        )

        self.assertEqual(
            [tensor.shape for tensor in (displacement, distance, weight, output)],
            [(edges, dimensions), (edges,), (edges,), (nodes, features)],
        )
        forbidden = {(nodes, nodes), (nodes, edges)}
        for tensor in (displacement, distance, weight, output):
            shapes = {
                tuple(int(size) for size in uop._shape)
                for uop in tensor.uop.toposort()
                if uop._shape is not None
            }
            self.assertTrue(forbidden.isdisjoint(shapes))
            self.assertLessEqual(
                max(prod(shape) for shape in shapes),
                max(edges * dimensions, nodes * features, nodes + 1),
            )

        programs = [call.src[0] for call in output.schedule_linear().src]
        self.assertEqual(len(programs), 4)
        self.assertEqual([program.arg.name for program in programs[:2]], ["edge_values"] * 2)
        self.assertEqual(programs[-1].arg.name, "csr_sum")
        ranges = [
            [uop for uop in program.toposort() if uop.op is Ops.RANGE]
            for program in programs
        ]
        for edge_projection in ranges[:2]:
            self.assertEqual(
                [(uop.src[0].arg, uop.arg[1]) for uop in edge_projection],
                [(edges * dimensions, AxisType.WEAK)],
            )
        self.assertEqual(
            [(uop.src[0].arg, uop.arg[1]) for uop in ranges[2]],
            [(edges, AxisType.WEAK), (dimensions, AxisType.REDUCE)],
        )
        self.assertEqual(
            (ranges[3][0].src[0].arg, ranges[3][0].arg[1]),
            (nodes * features, AxisType.WEAK),
        )
        self.assertEqual(ranges[3][1].arg[1], AxisType.WEAK)
        self.assertEqual(ranges[3][1].dtype, UOp.loop(-1).dtype)


if __name__ == "__main__":
    unittest.main()
