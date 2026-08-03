import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from tinygrad import Device

from experiments.mutag_data import observe
from tinymesh.datasets import mutag


ROWS = {
    "graph_indicator": "1\n1\n2\n2\n2\n",
    "node_labels": "0\n1\n2\n3\n6\n",
    "graph_labels": "-1\n1\n",
    "A": "1, 2\n2, 1\n3, 4\n4, 3\n4, 5\n5, 4\n",
    "edge_labels": "0\n0\n1\n1\n2\n2\n",
}


class MUTAGDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "MUTAG.zip"
        self._write()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_aligned_categorical_graphs(self) -> None:
        data = mutag(self.path, device=Device.DEFAULT)

        self.assertEqual(len(data), 2)
        self.assertEqual(data.node_types, ("C", "N", "O", "F", "I", "Cl", "Br"))
        self.assertEqual(data.bond_types, ("aromatic", "single", "double", "triple"))
        first = data[0]
        second = data[1]
        self.assertEqual(first[0].source, (0, 1))
        self.assertEqual(first[0].target, (1, 0))
        self.assertEqual(first[1].tolist(), [0, 1])
        self.assertEqual(first[2].tolist(), [0, 0])
        self.assertEqual(first[3], 0)
        self.assertEqual(second[0].source, (0, 1, 1, 2))
        self.assertEqual(second[0].target, (1, 0, 2, 1))
        self.assertEqual(second[1].tolist(), [2, 3, 6])
        self.assertEqual(second[2].tolist(), [1, 1, 2, 2])
        self.assertEqual(second[3], 1)
        self.assertEqual(first[1].one_hot(len(data.node_types)).float().shape, (2, 7))

    def test_witness_uses_the_public_loader(self) -> None:
        result = observe(self.path, device=Device.DEFAULT)

        self.assertEqual((result.graphs, result.nodes, result.directed_edges, result.bonds), (2, 5, 6, 3))
        self.assertEqual(result.class_counts, (1, 1))
        self.assertEqual(result.node_type_counts, (1, 1, 1, 1, 0, 0, 1))
        self.assertEqual(result.bond_type_counts, (2, 2, 2, 0))
        self.assertEqual(result.reciprocal_edges, result.directed_edges)

    def test_rejects_cross_graph_or_one_way_bonds(self) -> None:
        self._write(A="1, 3\n3, 1\n3, 4\n4, 3\n4, 5\n5, 4\n")
        with self.assertRaisesRegex(ValueError, "crosses graph"):
            mutag(self.path)

        self._write(A="1, 2\n3, 4\n4, 3\n4, 5\n5, 4\n", edge_labels="0\n1\n1\n2\n2\n")
        with self.assertRaisesRegex(ValueError, "both directions"):
            mutag(self.path)

    def test_bounds_decompressed_members(self) -> None:
        self._write(node_labels="0\n" * 5000)

        with self.assertRaisesRegex(ValueError, "node_labels exceeds"):
            mutag(self.path)

    def _write(self, **changes: str) -> None:
        rows = dict(ROWS, **changes)
        with ZipFile(self.path, "w") as archive:
            for name, values in rows.items():
                archive.writestr(f"MUTAG/MUTAG_{name}.txt", values)


if __name__ == "__main__":
    unittest.main()
