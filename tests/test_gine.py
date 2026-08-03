import unittest

from tinygrad import Device, Tensor

from experiments.gine import fit_one_step
from tinymesh import Graph
from tinymesh.nn import GINEConv


class GINETest(unittest.TestCase):
  def test_matches_edge_message_equation(self) -> None:
    graph = Graph(3, [0, 1, 2, 0], [1, 1, 1, 2])
    values = Tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], device=Device.DEFAULT).realize()
    edges = Tensor([[1.0], [2.0], [-1.0], [3.0]], device=Device.DEFAULT).realize()
    model = GINEConv(2, 1, 2, eps=0.5)
    model.edge.weight = Tensor([[2.0], [-1.0]], device=Device.DEFAULT).realize()
    model.edge.bias = Tensor([0.5, 1.0], device=Device.DEFAULT).realize()
    model.hidden.weight = Tensor.eye(2).to(Device.DEFAULT).realize()
    model.hidden.bias = Tensor.zeros(2, device=Device.DEFAULT).realize()
    model.output.weight = Tensor.eye(2).to(Device.DEFAULT).realize()
    model.output.bias = Tensor.zeros(2, device=Device.DEFAULT).realize()

    self.assertEqual(model(values, edges, graph).tolist(), [[1.5, 3.0], [21.0, 26.5], [18.0, 16.5]])

  def test_one_step_learns_from_edge_identity(self) -> None:
    observation = fit_one_step(Device.DEFAULT)

    self.assertAlmostEqual(observation.initial_loss, 5.0)
    self.assertEqual(observation.weight_gradient, (-11.0, -1.0))
    self.assertAlmostEqual(observation.final_loss, 0.9225, places=5)
    self.assertAlmostEqual(observation.reversed_edge_loss, 1.9225, places=5)
    self.assertAlmostEqual(observation.erased_edge_loss, 1.81, places=5)
    for actual, expected in zip(observation.hidden_weight, (0.55, 0.05)):
      self.assertAlmostEqual(actual, expected, places=6)

  def test_rejects_incompatible_shapes(self) -> None:
    graph = Graph(2, [0], [1])
    model = GINEConv(2, 1, 2)
    with self.assertRaisesRegex(ValueError, "values must have shape"):
      model(Tensor.ones(2, 1), Tensor.ones(1, 1), graph)
    with self.assertRaisesRegex(ValueError, "edge_values must have shape"):
      model(Tensor.ones(2, 2), Tensor.ones(1, 2), graph)


if __name__ == "__main__":
  unittest.main()
