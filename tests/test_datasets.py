import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tinygrad import Device

from tinymesh.datasets import chickenpox

SOURCE = {
    "edges": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "node_ids": {"RIGHT": 1, "LEFT": 0},
    "FX": [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ],
}


class ChickenpoxDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "chickenpox.json"
        self.path.write_text(json.dumps(SOURCE))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_lagged_fixed_graph_signal(self) -> None:
        dataset = chickenpox(self.path, lags=2, device=Device.DEFAULT)

        self.assertEqual(dataset.graph.nodes, 2)
        self.assertEqual(dataset.graph.source, (0, 0, 1, 1))
        self.assertEqual(dataset.graph.target, (0, 1, 0, 1))
        self.assertEqual(dataset.node_ids, ("LEFT", "RIGHT"))
        self.assertEqual(dataset.x.shape, (2, 2, 2))
        self.assertEqual(dataset.y.shape, (2, 2, 1))
        x0, y0 = dataset[0]
        x1, y1 = dataset[1]
        self.assertEqual(x0.tolist(), [[1.0, 2.0], [10.0, 20.0]])
        self.assertEqual(y0.tolist(), [[3.0], [30.0]])
        self.assertEqual(x1.tolist(), [[2.0, 3.0], [20.0, 30.0]])
        self.assertEqual(y1.tolist(), [[4.0], [40.0]])
        self.assertEqual(dataset.edge_weight.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_validates_source_contract(self) -> None:
        for lags in (0, 4):
            with self.assertRaises(ValueError):
                chickenpox(self.path, lags=lags)

        malformed = dict(SOURCE, node_ids={"LEFT": 1, "RIGHT": 2})
        self.path.write_text(json.dumps(malformed))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            chickenpox(self.path, lags=2)

        malformed = dict(SOURCE, FX=[[1.0], [2.0]])
        self.path.write_text(json.dumps(malformed))
        with self.assertRaisesRegex(ValueError, "node width"):
            chickenpox(self.path, lags=1)


if __name__ == "__main__":
    unittest.main()
