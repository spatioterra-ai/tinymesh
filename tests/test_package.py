import unittest
from pathlib import Path

import tinymesh
from tinygrad import Tensor
from tinymesh import Graph


class PackageTest(unittest.TestCase):
    def test_public_surface(self) -> None:
        self.assertEqual(tinymesh.__all__, ["Graph"])
        self.assertIs(tinymesh.Graph, Graph)
        self.assertFalse(hasattr(tinymesh, "CSRTopology"))

    def test_type_marker_and_runtime(self) -> None:
        self.assertTrue(Path(tinymesh.__file__).with_name("py.typed").is_file())
        self.assertEqual((Tensor([1, 2]) + 1).tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
