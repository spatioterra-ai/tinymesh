import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tinygrad import Device, Tensor, dtypes

from tinymesh.datasets import MontevideoBus, montevideo_bus

SOURCE = {
    "directed": True,
    "multigraph": False,
    "nodes": [
        {
            "bus_stop": 20,
            "lon": 1,
            "lat": 2,
            "X": {"y": [1, 2, 3]},
            "y": [2, 3, 4],
        },
        {
            "bus_stop": 10,
            "lon": 3.5,
            "lat": 4.5,
            "X": {"y": [10, 20, 30]},
            "y": [20, 30, 40],
        },
    ],
    "links": [
        {"source": 10, "target": 20, "weight": 7.5},
        {"source": 20, "target": 10, "weight": 8},
    ],
}


class MontevideoDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "montevideo.json"
        self.path.write_text(json.dumps(SOURCE))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lowers_raw_aligned_tensors(self) -> None:
        data = montevideo_bus(self.path, lags=2, device=Device.DEFAULT)
        signal = data.signal

        self.assertEqual(signal.graph.nodes, 2)
        self.assertEqual(signal.graph.source, (1, 0))
        self.assertEqual(signal.graph.target, (0, 1))
        self.assertEqual(signal.node_ids, ("20", "10"))
        self.assertEqual(signal.x.shape, (1, 2, 2))
        self.assertEqual(signal.y.shape, (1, 2, 1))
        self.assertEqual(signal.x.tolist(), [[[1.0, 2.0], [10.0, 20.0]]])
        self.assertEqual(signal.y.tolist(), [[[4.0], [40.0]]])
        self.assertIsNone(signal.edge_weight)
        self.assertEqual(data.position.tolist(), [[1.0, 2.0], [3.5, 4.5]])
        self.assertEqual(data.road_distance.tolist(), [7.5, 8.0])
        self.assertEqual(data.coordinate_frame, "EPSG:32721")
        self.assertEqual(data.length_unit, "m")
        self.assertEqual(data.position.dtype, signal.x.dtype)
        self.assertEqual(data.position.device, signal.x.device)
        self.assertEqual(data.road_distance.dtype, signal.x.dtype)
        self.assertEqual(data.road_distance.device, signal.x.device)

    def test_composes_coordinate_distance_in_edge_order(self) -> None:
        data = montevideo_bus(self.path, lags=1, device=Device.DEFAULT)
        source = data.signal.graph.edge_values(data.position, endpoint="source")
        target = data.signal.graph.edge_values(data.position, endpoint="target")
        distance = ((target - source) ** 2).sum(axis=-1).sqrt().realize()

        for actual in distance.tolist():
            self.assertAlmostEqual(actual, math.sqrt(12.5), places=6)
        self.assertEqual(data.road_distance.tolist(), [7.5, 8.0])

    def test_validates_lag_count(self) -> None:
        for lags in (0, True, 3):
            with self.subTest(lags=lags), self.assertRaisesRegex(ValueError, "lags"):
                montevideo_bus(self.path, lags=lags)

    def test_record_rejects_misaligned_tensors(self) -> None:
        data = montevideo_bus(self.path, lags=1, device=Device.DEFAULT)

        with self.assertRaisesRegex(ValueError, "position"):
            MontevideoBus(data.signal, data.position[:1], data.road_distance)
        with self.assertRaisesRegex(ValueError, "road_distance"):
            MontevideoBus(data.signal, data.position, data.road_distance[:1])
        with self.assertRaisesRegex(ValueError, "dtype and device"):
            MontevideoBus(data.signal, data.position.cast(dtypes.float16), data.road_distance)
        edge_weight = Tensor.ones(data.signal.graph.edges, dtype=data.signal.x.dtype, device=data.signal.x.device)
        with self.assertRaisesRegex(ValueError, "edge_weight"):
            MontevideoBus(replace(data.signal, edge_weight=edge_weight), data.position, data.road_distance)


if __name__ == "__main__":
    unittest.main()
