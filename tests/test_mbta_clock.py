import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.mbta_clock import (
  EXPECTED_CANDIDATES,
  FIXTURE,
  ClockError,
  admissible,
  audit_lane,
  bin_of,
  observe,
  retains_target,
)


class MbtaClockTest(unittest.TestCase):
  def test_half_open_bins_and_target_separation(self) -> None:
    self.assertEqual((bin_of(29, 30), bin_of(30, 30)), (0, 1))
    self.assertTrue(retains_target(29, 30, 30))
    self.assertFalse(retains_target(30, 31, 30))
    with self.assertRaisesRegex(ClockError, "seconds must be positive"):
      bin_of(0, 0)
    with self.assertRaisesRegex(ClockError, "time did not advance"):
      retains_target(30, 30, 30)

  def test_lane_audit_separates_collision_equal_time_and_empty_cells(self) -> None:
    result = audit_lane((0, 29, 30, 31, 31, 91), 30)
    self.assertEqual((result.cells, result.occupied_cells, result.empty_cells), (4, 3, 1))
    self.assertEqual((result.causal_collision_bins, result.causal_collision_events), (2, 5))
    self.assertEqual((result.equal_time_sets, result.equal_time_events), (1, 2))
    self.assertEqual(result.maximum_events_per_cell, 3)

  def test_lane_day_boundaries_are_independent(self) -> None:
    self.assertEqual(audit_lane((0, 60), 30), audit_lane((86_400, 86_460), 30))
    with self.assertRaisesRegex(ClockError, "has no events"):
      audit_lane((), 30)

  def test_admission_requires_identity_and_zero_causal_collisions(self) -> None:
    self.assertTrue(admissible(True, 0))
    self.assertFalse(admissible(True, 1))
    self.assertFalse(admissible(False, 0))

  def test_retained_full_population_decision_is_exact(self) -> None:
    result = observe()
    self.assertEqual((result.physical_departures, result.targets, result.lanes), (947_489, 940_551, 259))
    self.assertEqual(result.candidates, EXPECTED_CANDIDATES)
    self.assertEqual(result.decision, "stop:no_identity_preserving_clock")
    self.assertEqual(result.stage_1_consequence, "close:stage_1_snapshot")
    self.assertEqual(result, observe())

  def test_retained_audit_drift_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      audit_path = fixture / "audit.json"
      audit = json.loads(audit_path.read_text())
      audit["candidates"][0]["targets"] += 1
      audit_path.write_text(json.dumps(audit))
      with self.assertRaisesRegex(ClockError, "target accounting drift"):
        observe(fixture)


if __name__ == "__main__":
  unittest.main()
