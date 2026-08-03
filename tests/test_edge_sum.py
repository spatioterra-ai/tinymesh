import unittest
from math import prod

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from tinymesh import Graph


SOURCE = [0, 1, 2, 1]
TARGET = [2, 2, 3, 0]
VALUES = [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]
EXPECTED = [[13.0, 17.0], [0.0, 0.0], [4.0, 7.0], [7.0, 11.0]]


class EdgeSumTest(unittest.TestCase):
  def test_sums_coo_values_at_targets(self) -> None:
    graph = Graph(4, SOURCE, TARGET)
    values = Tensor(VALUES, device=Device.DEFAULT).realize()

    self.assertEqual(graph.sum_edges(values).tolist(), EXPECTED)

  def test_gradient_returns_each_target_row_to_its_edges(self) -> None:
    graph = Graph(4, SOURCE, TARGET)
    values = Tensor(VALUES, device=Device.DEFAULT).realize()
    gradient = Tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0]], device=Device.DEFAULT)

    actual = graph.sum_edges(values).gradient(values, gradient=gradient)[0]

    self.assertEqual(actual.tolist(), [[11.0, 13.0], [11.0, 13.0], [17.0, 19.0], [2.0, 3.0]])

  def test_leading_axes_share_one_sparse_call(self) -> None:
    graph = Graph(4, SOURCE, TARGET)
    values = Tensor([VALUES, [[2 * value for value in row] for row in VALUES]], device=Device.DEFAULT).realize()
    output = graph.sum_edges(values)

    calls = [uop for uop in output.uop.toposort() if uop.op is Ops.CALL]
    self.assertEqual(output.tolist(), [EXPECTED, [[2 * value for value in row] for row in EXPECTED]])
    self.assertEqual(len(calls), 1)
    limit = max(graph.nodes * 4, graph.edges, graph.nodes + 1)
    self.assertTrue(all(
      prod(int(size) for size in uop._shape) <= limit
      for uop in calls[0].src[0].toposort()
      if uop._shape is not None
    ))

  def test_empty_edges_preserve_shape_and_gradient(self) -> None:
    graph = Graph(3, [], [])
    values = Tensor.zeros(2, 0, 4, device=Device.DEFAULT).realize()
    output = graph.sum_edges(values)
    gradient = output.sum().gradient(values)[0]

    self.assertEqual(output.shape, (2, 3, 4))
    self.assertEqual(output.tolist(), [[[0.0] * 4] * 3] * 2)
    self.assertEqual(gradient.shape, values.shape)

  def test_rejects_incompatible_values(self) -> None:
    graph = Graph(3, [0], [1])
    with self.assertRaisesRegex(ValueError, r"shape \[\.\.\., E, H\]"):
      graph.sum_edges(Tensor.ones(1, device=Device.DEFAULT))
    with self.assertRaisesRegex(ValueError, "1 edge rows"):
      graph.sum_edges(Tensor.ones(2, 1, device=Device.DEFAULT))


if __name__ == "__main__":
  unittest.main()
