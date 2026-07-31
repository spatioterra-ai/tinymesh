import unittest

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from tinymesh import Graph
from tinymesh.nn import A3TGCN


GRAPH = Graph(3, [0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0])


class A3TGCNTest(unittest.TestCase):
    def test_attention_weights_independent_period_encodings(self) -> None:
        model = A3TGCN(1, 2, periods=3)
        model.attention.weight = Tensor([-1.0, 0.5, 2.0], device=Device.DEFAULT).realize()
        values = Tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[3.0], [2.0], [1.0]],
                [[2.0], [4.0], [1.0]],
            ],
            device=Device.DEFAULT,
        ).realize()

        probability = model.attention.weight.softmax(axis=0)
        expected = sum(
            (
                model.cell(values[period], GRAPH) * probability[period]
                for period in range(3)
            ),
        )
        actual = model(values, GRAPH)

        for actual_row, expected_row in zip(actual.tolist(), expected.tolist()):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_batch_axis_matches_independent_examples(self) -> None:
        model = A3TGCN(1, 2, periods=2)
        values = Tensor(
            [
                [[[1.0], [2.0], [3.0]], [[3.0], [2.0], [1.0]]],
                [[[2.0], [1.0], [0.0]], [[0.0], [1.0], [2.0]]],
            ],
            device=Device.DEFAULT,
        ).realize()

        expected = Tensor.stack(*(model(example, GRAPH) for example in values))
        actual = model(values, GRAPH)
        for actual_batch, expected_batch in zip(actual.tolist(), expected.tolist()):
            for actual_row, expected_row in zip(actual_batch, expected_batch):
                for actual_value, expected_value in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_each_period_reuses_one_sparse_tgcn_projection(self) -> None:
        output = A3TGCN(1, 2, periods=3)(
            Tensor.zeros(1, 3, 3, 1, device=Device.DEFAULT),
            GRAPH,
        )
        calls = [
            uop
            for uop in output.uop.toposort()
            if uop.op is Ops.CALL and uop.src[0].arg.name == "csr_sum"
        ]
        self.assertEqual(len(calls), 3)

    def test_attention_and_graph_parameters_receive_gradients(self) -> None:
        model = A3TGCN(1, 2, periods=2)
        values = Tensor(
            [[[1.0], [2.0], [3.0]], [[3.0], [1.0], [2.0]]],
            device=Device.DEFAULT,
        ).realize()

        model(values, GRAPH).square().sum().backward()

        self.assertIsNotNone(model.attention.weight.grad)
        self.assertIsNotNone(model.cell.graph_projection.linear.weight.grad)
        self.assertGreater(float(model.attention.weight.grad.abs().sum().item()), 0)

    def test_rejects_invalid_periods_or_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "periods must be positive"):
            A3TGCN(1, 2, periods=0)

        model = A3TGCN(1, 2, periods=2)
        with self.assertRaisesRegex(ValueError, r"shape \[\.\.\., 2, 3, 1\]"):
            model(Tensor.zeros(3, 1, device=Device.DEFAULT), GRAPH)


if __name__ == "__main__":
    unittest.main()
