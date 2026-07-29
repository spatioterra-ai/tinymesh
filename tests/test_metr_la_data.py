import csv
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from tinygrad import Device

from experiments.metr_la_data import observe
from tinymesh.datasets import metr_la


SENSOR_IDS = ("a", "b", "c")
TRAFFIC = (
    ("2012-03-01 00:00:00", 10, 0, 30),
    ("2012-03-01 00:05:00", 11, 21, 31),
    ("2012-03-01 00:10:00", 12, 22, 0),
)
DISTANCES = (
    ("a", "a", 0),
    ("a", "b", 1),
    ("b", "a", 2),
    ("b", "b", 0),
    ("b", "c", 4),
    ("c", "a", 10),
    ("c", "c", 0),
    ("outside", "a", 1),
)


class METRLADatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.joinpath("graph_sensor_ids.txt").write_text(",".join(SENSOR_IDS))
        self._write_traffic(("", *SENSOR_IDS), TRAFFIC)
        self._write_distances(DISTANCES)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_aligns_raw_time_nodes_missingness_and_affinity(self) -> None:
        data = metr_la(self.root, device=Device.DEFAULT)

        self.assertEqual(data.sensor_ids, SENSOR_IDS)
        self.assertEqual(data.timestamps, (
            datetime(2012, 3, 1, 0, 0),
            datetime(2012, 3, 1, 0, 5),
            datetime(2012, 3, 1, 0, 10),
        ))
        self.assertEqual(data.sample_minutes, 5)
        self.assertEqual(data.speed.shape, (3, 3))
        self.assertEqual(data.speed.tolist(), [[10.0, 0.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 0.0]])
        self.assertEqual(data.observed.tolist(), [[True, False, True], [True, True, True], [True, True, False]])
        self.assertEqual(data.graph.source, (0, 0, 1, 1, 1, 2))
        self.assertEqual(data.graph.target, (0, 1, 0, 1, 2, 2))
        self.assertEqual(data.affinity.shape, (6,))
        self.assertAlmostEqual(data.affinity[1].item(), 0.9159316)
        self.assertAlmostEqual(data.affinity[4].item(), 0.2453627)

    def test_witness_uses_the_public_loader(self) -> None:
        result = observe(self.root, device=Device.DEFAULT)

        self.assertEqual((result.nodes, result.edges, result.steps), (3, 6, 3))
        self.assertEqual((result.values, result.observed_values, result.missing_values), (9, 7, 2))
        self.assertEqual((result.self_loops, result.asymmetric_edges), (3, 1))

    def test_rejects_misaligned_or_irregular_traffic(self) -> None:
        self._write_traffic(("", "b", "a", "c"), TRAFFIC)
        with self.assertRaisesRegex(ValueError, "sensor ID order"):
            metr_la(self.root)

        self._write_traffic(("", *SENSOR_IDS), (*TRAFFIC[:2], ("2012-03-01 00:11:00", 12, 22, 32)))
        with self.assertRaisesRegex(ValueError, "five minutes"):
            metr_la(self.root)

    def test_rejects_ambiguous_or_incomplete_graphs(self) -> None:
        self._write_distances((*DISTANCES, ("a", "b", 3)))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            metr_la(self.root)

        self._write_distances(tuple(row for row in DISTANCES if row[:2] != ("c", "c")))
        with self.assertRaisesRegex(ValueError, "zero-distance self edge"):
            metr_la(self.root)

    def _write_traffic(self, header: tuple[object, ...], rows: tuple[tuple[object, ...], ...]) -> None:
        with self.root.joinpath("METR-LA.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def _write_distances(self, rows: tuple[tuple[object, ...], ...]) -> None:
        with self.root.joinpath("distances_la_2012.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("from", "to", "cost"))
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
