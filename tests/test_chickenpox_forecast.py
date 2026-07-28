import unittest

from tinygrad import Device, Tensor

from experiments.chickenpox_forecast import Forecast, LSTMForecast, _without_self_loops
from experiments.gconv_gru import GConvGRU
from experiments.tgcn import TGCN
from tinymesh import Graph


class ChickenpoxForecastTest(unittest.TestCase):
    def test_recurrent_forecasts_accept_batched_windows(self) -> None:
        graph = Graph(3, [0, 1, 1, 2], [1, 0, 2, 1])
        values = Tensor.ones(2, 4, 3, 1, device=Device.DEFAULT).realize()

        for cell in (TGCN(1, 2), GConvGRU(1, 2, 2)):
            self.assertEqual(Forecast(cell, 2)(values, graph).shape, (2, 3, 1))
        self.assertEqual(LSTMForecast(2)(values, graph).shape, (2, 3, 1))

    def test_self_loop_removal_preserves_edge_order(self) -> None:
        graph = Graph(3, [0, 0, 1, 2, 2], [0, 1, 0, 2, 1])

        loop_free = _without_self_loops(graph)

        self.assertEqual(loop_free.source, (0, 1, 2))
        self.assertEqual(loop_free.target, (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
