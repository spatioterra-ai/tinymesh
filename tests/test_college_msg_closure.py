import unittest

from experiments.college_msg_closure import observe
from tinymesh import TemporalEdges


class CollegeMsgClosureTest(unittest.TestCase):
  def test_measurement_uses_prior_time_and_exact_pair_exposure(self) -> None:
    result = observe(TemporalEdges(
      4,
      (0, 2, 1, 0, 2, 3),
      (1, 3, 2, 2, 0, 3),
      (10, 10, 20, 30, 30, 40),
    ))

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
    self.assertEqual(result.decision, "retain:closure_research_only")

  def test_same_timestamp_edges_do_not_close_each_other(self) -> None:
    result = observe(TemporalEdges(4, (0, 1, 0, 2), (1, 2, 2, 3), (10, 10, 10, 20)))

    self.assertEqual(result.closure.entry_contacts, 4)
    self.assertEqual(result.closure.wedge_formations, 0)

  def test_rejects_empty_stream(self) -> None:
    with self.assertRaisesRegex(ValueError, "at least one"):
      observe(TemporalEdges(1, (), (), ()))


if __name__ == "__main__":
  unittest.main()
