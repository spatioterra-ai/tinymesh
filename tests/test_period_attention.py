import unittest

from tinygrad import Device, Tensor, nn

from tinymesh.nn import PeriodAttention


class PeriodAttentionTest(unittest.TestCase):
    def test_matches_weighted_reference(self) -> None:
        attention = PeriodAttention(3)
        attention.weight = Tensor([-1.0, 0.5, 2.0], device=Device.DEFAULT).realize()
        states = (
            Tensor([[1.0, 2.0], [3.0, 4.0]], device=Device.DEFAULT),
            Tensor([[2.0, 0.0], [1.0, 3.0]], device=Device.DEFAULT),
            Tensor([[4.0, 1.0], [2.0, 0.0]], device=Device.DEFAULT),
        )
        probability = attention.weight.softmax(axis=0)
        expected = sum(
            (state * probability[period] for period, state in enumerate(states[1:], 1)),
            start=states[0] * probability[0],
        )

        self.assertTrue(attention(*states).allclose(expected).item())

    def test_weight_is_one_tinygrad_parameter(self) -> None:
        attention = PeriodAttention(2)
        states = (
            Tensor([[1.0], [2.0]], device=Device.DEFAULT),
            Tensor([[3.0], [1.0]], device=Device.DEFAULT),
        )

        attention(*states).square().sum().backward()

        self.assertEqual(nn.state.get_parameters(attention), [attention.weight])
        self.assertIsNotNone(attention.weight.grad)
        self.assertGreater(attention.weight.grad.abs().sum().item(), 0)

    def test_rejects_invalid_periods_count_or_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "periods must be positive"):
            PeriodAttention(0)

        attention = PeriodAttention(2)
        with self.assertRaisesRegex(ValueError, "expected 2 period states, got 1"):
            attention(Tensor.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "share one shape"):
            attention(Tensor.zeros(2, 1), Tensor.zeros(3, 1))


if __name__ == "__main__":
    unittest.main()
