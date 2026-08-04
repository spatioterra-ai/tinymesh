import unittest

from experiments.framework_benchmark import _edges, _values


class FrameworkBenchmarkTest(unittest.TestCase):
    def test_balanced_edges_and_explicit_self_loops(self) -> None:
        source, target = _edges(4, 2)
        recurrent_source, recurrent_target = _edges(4, 2, self_loops=True)

        self.assertEqual(source, [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(target, [1, 2, 2, 3, 3, 0, 0, 1])
        self.assertEqual(recurrent_source[-4:], [0, 1, 2, 3])
        self.assertEqual(recurrent_target[-4:], [0, 1, 2, 3])

    def test_values_are_deterministic_and_bounded(self) -> None:
        self.assertEqual(_values(1, 3), [-15 / 31, -14 / 31, -13 / 31])
        self.assertEqual(_values(1, 3, offset=31), _values(1, 3))
        self.assertTrue(all(-0.5 < value < 0.5 for value in _values(3, 4)))


if __name__ == "__main__":
    unittest.main()
