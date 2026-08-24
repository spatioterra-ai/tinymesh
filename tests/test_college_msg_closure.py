import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.college_msg_closure import Interaction, load, observe, parse


class CollegeMsgClosureTest(unittest.TestCase):
  def test_parse_preserves_direction_identity_and_time(self) -> None:
    interactions = parse(["1 2 10\n", "2 1 10\n", "2 3 20\n"])

    self.assertEqual(
      interactions,
      (
        Interaction(10, 1, 2),
        Interaction(10, 2, 1),
        Interaction(20, 2, 3),
      ),
    )

  def test_parse_rejects_bad_shape_value_and_order(self) -> None:
    for lines, message in (
      (["1 2\n"], "expected"),
      (["one 2 3\n"], "integers"),
      (["1 -2 3\n"], "non-negative"),
      (["1 2 20\n", "2 3 10\n"], "backward"),
      ([], "no interactions"),
    ):
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        parse(lines)

  def test_measurement_uses_prior_time_and_exact_pair_exposure(self) -> None:
    result = observe(parse([
      "1 2 10\n",
      "3 4 10\n",
      "2 3 20\n",
      "1 3 30\n",
      "3 1 30\n",
      "4 4 40\n",
    ]))

    self.assertEqual((result.source.messages, result.source.nodes, result.source.self_messages), (6, 4, 1))
    self.assertEqual((result.source.directed_pairs, result.source.undirected_pairs), (5, 4))
    self.assertEqual(
      (
        result.closure.first_contacts,
        result.closure.entry_contacts,
        result.closure.repeat_messages,
        result.closure.wedge_formations,
        result.closure.non_wedge_formations,
      ),
      (4, 2, 1, 1, 1),
    )
    self.assertEqual((result.closure.wedge_pair_seconds, result.closure.non_wedge_pair_seconds), (40, 50))
    self.assertEqual(result.projection, "undirected_first_contact")

  def test_same_timestamp_edges_do_not_close_each_other(self) -> None:
    result = observe(parse(["1 2 10\n", "2 3 10\n", "1 3 10\n", "3 4 20\n"]))

    self.assertEqual(result.closure.entry_contacts, 4)
    self.assertEqual(result.closure.wedge_formations, 0)

  def test_load_rejects_artifact_drift_before_parsing(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "source.gz"
      with gzip.open(path, "wt", encoding="ascii") as target:
        target.write("1 2 10\n")
      payload = path.read_bytes()

      with patch("experiments.college_msg_closure.SOURCE_BYTES", len(payload)), patch(
        "experiments.college_msg_closure.SOURCE_SHA256", hashlib.sha256(payload).hexdigest()
      ):
        self.assertEqual(load(path), (Interaction(10, 1, 2),))
      with self.assertRaisesRegex(ValueError, "pinned CollegeMsg"):
        load(path)


if __name__ == "__main__":
  unittest.main()
