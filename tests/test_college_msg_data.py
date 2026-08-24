import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinymesh import TemporalEdges
from tinymesh.datasets import CollegeMsg, college_msg


class CollegeMsgDataTest(unittest.TestCase):
  def test_preserves_messages_and_compacts_source_identity(self) -> None:
    dataset = self.load("10 30 100\n30 10 100\n20 10 200\n20 10 200\n")

    self.assertIsInstance(dataset, CollegeMsg)
    self.assertEqual(dataset.node_ids, (10, 20, 30))
    self.assertEqual(dataset.events.nodes, 3)
    self.assertEqual(dataset.events.edges, 4)
    self.assertEqual(dataset.events.source, (0, 2, 1, 1))
    self.assertEqual(dataset.events.target, (2, 0, 0, 0))
    self.assertEqual(dataset.events.timestamp, (100, 100, 200, 200))

  def test_rejects_artifact_drift_before_parsing(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "CollegeMsg.txt.gz"
      with gzip.open(path, "wt", encoding="ascii") as target:
        target.write("1 2 10\n")

      with self.assertRaisesRegex(ValueError, "identity mismatch"):
        college_msg(path)

  def test_rejects_malformed_rows(self) -> None:
    for text, message in (
      ("1 2\n", "source, target, timestamp"),
      ("one 2 3\n", "integers"),
      ("1 -2 3\n", "non-negative"),
      ("1 2 20\n2 3 10\n", "moved backward"),
      ("", "at least one"),
    ):
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        self.load(text)

  def test_dataset_requires_one_unique_source_identity_per_node(self) -> None:
    events = TemporalEdges(2, (), (), ())

    with self.assertRaisesRegex(ValueError, "expected 2"):
      CollegeMsg(events, (10,))
    with self.assertRaisesRegex(ValueError, "unique"):
      CollegeMsg(events, (10, 10))
    with self.assertRaisesRegex(ValueError, "non-negative integers"):
      CollegeMsg(events, (10, -1))

  @staticmethod
  def load(text: str) -> CollegeMsg:
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "CollegeMsg.txt.gz"
    with gzip.open(path, "wt", encoding="ascii") as target:
      target.write(text)
    payload = path.read_bytes()
    source = (path.name, "unused", len(payload), hashlib.sha256(payload).hexdigest())
    with temporary, patch("tinymesh.datasets._COLLEGE_MSG_SOURCE", source):
      return college_msg(path)


if __name__ == "__main__":
  unittest.main()
