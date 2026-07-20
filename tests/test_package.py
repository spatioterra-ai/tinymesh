import unittest

import tinymesh
from tinygrad import Tensor


class PackageTest(unittest.TestCase):
    def test_runtime_imports(self) -> None:
        self.assertEqual(tinymesh.__name__, "tinymesh")
        self.assertEqual((Tensor([1, 2]) + 1).tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
