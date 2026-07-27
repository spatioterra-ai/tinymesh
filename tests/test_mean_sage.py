import unittest

from tinygrad import Device, Tensor, dtypes

from experiments.csr_aggregation import CSRTopology
from experiments.mean_sage import MeanSAGE, fit_one_step


class MeanSAGETest(unittest.TestCase):
    def test_reuses_fixed_degree_normalization(self) -> None:
        topology = CSRTopology(4, [0, 1, 1], [2, 2, 3])

        self.assertIs(
            topology._inverse_degree(Device.DEFAULT, dtypes.float),
            topology._inverse_degree(Device.DEFAULT, dtypes.float),
        )

    def test_mean_neighbor_and_root_paths(self) -> None:
        topology = CSRTopology(4, [0, 1, 1], [2, 2, 3])
        values = Tensor([[2.0], [4.0], [8.0], [16.0]], device=Device.DEFAULT).realize()
        model = MeanSAGE(1, 1)
        model.root.weight = Tensor([[3.0]], device=Device.DEFAULT).realize()
        model.neighbor.weight = Tensor([[5.0]], device=Device.DEFAULT).realize()

        self.assertEqual(model(values, topology).tolist(), [[6.0], [12.0], [39.0], [68.0]])

    def test_optimizer_reaches_neighbor_parameter_through_csr_backward(self) -> None:
        observation = fit_one_step(Device.DEFAULT)

        self.assertAlmostEqual(observation.initial_loss, 1.0)
        self.assertAlmostEqual(observation.neighbor_gradient, -2.0)
        self.assertAlmostEqual(observation.final_loss, 0.0)
        self.assertAlmostEqual(observation.root_weight, 0.0)
        self.assertAlmostEqual(observation.neighbor_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
