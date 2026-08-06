import unittest

from tinygrad import Tensor

from experiments.mutag_protocol import nearest_label_accuracy


class MUTAGRetrievalTest(unittest.TestCase):
  def test_exact_cosine_search_uses_only_training_neighbors(self) -> None:
    features = Tensor([
      [1.0, 0.0, 2.0],
      [0.9, 0.1, 2.0],
      [0.0, 1.0, 2.0],
      [0.1, 0.9, 2.0],
    ])

    self.assertEqual(nearest_label_accuracy(features, (0, 0, 1, 1), (0, 2), (1, 3)), 1.0)


if __name__ == "__main__":
  unittest.main()
