import unittest

from tinygrad import Device, Tensor

from experiments.mean_sage import MeanSAGE, fit_one_step
from tinymesh import Graph


class MeanSAGETest(unittest.TestCase):
    def test_mean_neighbor_and_root_paths(self) -> None:
        graph = Graph(4, [0, 1, 1], [2, 2, 3])
        values = Tensor([[2.0], [4.0], [8.0], [16.0]], device=Device.DEFAULT).realize()
        model = MeanSAGE(1, 1)
        model.root.weight = Tensor([[3.0]], device=Device.DEFAULT).realize()
        model.neighbor.weight = Tensor([[5.0]], device=Device.DEFAULT).realize()

        self.assertEqual(model(values, graph).tolist(), [[6.0], [12.0], [39.0], [68.0]])

    def test_optimizer_reaches_neighbor_parameter_through_csr_backward(self) -> None:
        observation = fit_one_step(Device.DEFAULT)

        self.assertAlmostEqual(observation.initial_loss, 1.0)
        self.assertAlmostEqual(observation.neighbor_gradient, -2.0)
        self.assertAlmostEqual(observation.final_loss, 0.0)
        self.assertAlmostEqual(observation.root_weight, 0.0)
        self.assertAlmostEqual(observation.neighbor_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
