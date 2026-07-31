import unittest
from importlib.metadata import requires
from pathlib import Path

import tinymesh
from tinygrad import Tensor
from tinymesh import Graph, StaticGraphTemporalSignal


class PackageTest(unittest.TestCase):
    def test_public_surface(self) -> None:
        self.assertEqual(tinymesh.__all__, ["Graph", "StaticGraphTemporalSignal"])
        self.assertIs(tinymesh.Graph, Graph)
        self.assertIs(tinymesh.StaticGraphTemporalSignal, StaticGraphTemporalSignal)
        self.assertFalse(hasattr(tinymesh, "CSRTopology"))

    def test_nn_surface_is_direct(self) -> None:
        from tinymesh import nn

        self.assertEqual(
            {name for name in vars(nn) if name.endswith("Conv")},
            {"ChebConv", "GATConv", "GCNConv", "SAGEConv"},
        )
        self.assertTrue(all(hasattr(nn, name) for name in (
            "A3TGCN",
            "DiffusionGRU",
            "DirectedDiffusion",
            "GConvGRU",
            "PeriodAttention",
            "TGCN",
        )))

    def test_type_marker_and_runtime(self) -> None:
        self.assertTrue(Path(tinymesh.__file__).with_name("py.typed").is_file())
        self.assertEqual((Tensor([1, 2]) + 1).tolist(), [2, 3])

    def test_tinygrad_is_the_only_runtime_dependency(self) -> None:
        dependencies = [
            requirement.split()[0]
            for requirement in requires("tinymesh") or []
        ]
        self.assertEqual(dependencies, ["tinygrad"])


if __name__ == "__main__":
    unittest.main()
