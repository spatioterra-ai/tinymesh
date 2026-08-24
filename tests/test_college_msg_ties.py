import unittest

from experiments.college_msg_ties import observe
from tinymesh import TemporalEdges


class CollegeMsgTiesTest(unittest.TestCase):
  def test_separates_strength_reciprocity_overlap_and_fragmentation(self) -> None:
    interactions = TemporalEdges(
      6,
      (0, 1, 0, 2, 2, 2, 2, 3, 3, 3, 4, 0, 1),
      (1, 0, 2, 1, 1, 3, 2, 3, 4, 5, 5, 1, 0),
      (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 86_400, 86_401),
    )
    result = observe(interactions, seed=7)

    self.assertEqual((result.messages, result.self_messages, result.ties), (13, 2, 7))
    self.assertEqual((result.reciprocal_ties, result.local_bridges, result.undefined_overlaps), (1, 1, 0))
    self.assertEqual([group.label for group in result.strength[0].groups], ["1", "2-3", "4-7"])
    self.assertEqual([group.edges for group in result.strength[0].groups], [5, 1, 1])
    self.assertEqual([group.label for group in result.strength[1].groups], ["1", "2-3"])
    self.assertEqual([group.edges for group in result.reciprocity.groups], [6, 1])
    self.assertEqual(result.reciprocity.groups[1].mean_overlap, 1.0)

    for audit in result.fragmentation:
      for curve in (audit.weak_first, audit.strong_first):
        self.assertEqual((curve.points[0].largest_component, curve.points[0].components), (6, 1))
        self.assertEqual((curve.points[-1].largest_component, curve.points[-1].components), (1, 6))
    self.assertEqual(result.random_baseline.points[-1].removed_edges, 7)
    self.assertEqual(result, observe(interactions, seed=7))

  def test_preserves_unavailable_overlap_and_empty_reciprocity_group(self) -> None:
    result = observe(TemporalEdges(2, (0,), (1,), (0,)))

    self.assertEqual(result.undefined_overlaps, 1)
    self.assertIsNone(result.strength[0].groups[0].mean_overlap)
    self.assertEqual(result.reciprocity.groups[1].edges, 0)
    self.assertIsNone(result.reciprocity.groups[1].mean_embeddedness)

  def test_rejects_invalid_seed_and_stream_without_contacts(self) -> None:
    with self.assertRaisesRegex(ValueError, "seed"):
      observe(TemporalEdges(1, (), (), ()), seed=-1)
    with self.assertRaisesRegex(ValueError, "non-self"):
      observe(TemporalEdges(1, (0,), (0,), (0,)))


if __name__ == "__main__":
  unittest.main()
