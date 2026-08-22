import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from experiments.mbta_headway_task import (
  FIXTURE,
  Event,
  Lane,
  TaskError,
  causal_prefix,
  observe,
  split_for,
  strict_target,
)


class MbtaHeadwayTaskTest(unittest.TestCase):
  def test_strict_target_and_causal_prefix_exclude_label(self) -> None:
    lane = Lane("place-davis", "Red", 0)
    day = date(2026, 8, 11)
    events = (
      Event(day, 100, "a", lane),
      Event(day, 160, "b", lane),
      Event(day, 160, "c", lane),
      Event(day, 230, "d", lane),
    )

    target = strict_target(events, events[0], events[1])

    self.assertEqual((target.cutoff, target.seconds), (101, 60))
    self.assertEqual(causal_prefix(events, target), (events[0],))
    self.assertNotIn(target.target, causal_prefix(events, target))

  def test_non_next_cross_lane_duplicate_and_out_of_range_fail(self) -> None:
    lane = Lane("place-davis", "Red", 0)
    other = Lane("place-davis", "Red", 1)
    day = date(2026, 8, 11)
    events = (
      Event(day, 100, "a", lane),
      Event(day, 160, "b", lane),
      Event(day, 230, "c", lane),
      Event(day, 160, "d", other),
    )

    with self.assertRaisesRegex(TaskError, "not the next strict"):
      strict_target(events, events[0], events[2])
    with self.assertRaisesRegex(TaskError, "share one service-day lane"):
      strict_target(events, events[0], events[3])
    with self.assertRaisesRegex(TaskError, "unique physical identities"):
      strict_target(events + (events[0],), events[0], events[1])
    with self.assertRaisesRegex(TaskError, "outside frozen population"):
      split_for(date(2026, 7, 23))

  def test_temporal_split_boundaries_are_exact(self) -> None:
    self.assertEqual(split_for(date(2026, 7, 24)), "train")
    self.assertEqual(split_for(date(2026, 8, 10)), "train")
    self.assertEqual(split_for(date(2026, 8, 11)), "validation")
    self.assertEqual(split_for(date(2026, 8, 15)), "validation")
    self.assertEqual(split_for(date(2026, 8, 16)), "test")
    self.assertEqual(split_for(date(2026, 8, 20)), "test")

  def test_retained_validation_and_single_test_open_are_exact(self) -> None:
    result = observe()

    self.assertEqual(
      (result.targets, result.train_targets, result.validation_targets, result.test_targets),
      (940_551, 625_073, 138_910, 176_568),
    )
    self.assertEqual(
      (result.split, result.temporal_bin_hours, result.temporal_minimum_support),
      ("validation", 1, 4),
    )
    self.assertEqual(
      (result.persistence_mae_seconds, result.temporal_mae_seconds, result.plan_mae_seconds),
      (217.968544, 157.151623, 152.647955),
    )
    self.assertEqual((result.best_baseline, result.decision), ("plan", "freeze:open_test_once"))
    self.assertEqual(result, observe())

    test = observe(test=True)
    self.assertEqual((test.split, test.temporal_bin_hours, test.temporal_minimum_support), ("test", 1, 4))
    self.assertEqual(
      (test.persistence_mae_seconds, test.temporal_mae_seconds, test.plan_mae_seconds),
      (229.801565, 159.787552, 166.062206),
    )
    self.assertEqual((test.best_baseline, test.decision), ("temporal", "freeze:stage_4_input"))

  def test_retained_artifacts_fail_closed_on_drift(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      protocol_path = fixture / "protocol.json"
      protocol = json.loads(protocol_path.read_text())
      protocol["target_counts"][0]["targets"] += 1
      protocol_path.write_text(json.dumps(protocol))

      with self.assertRaisesRegex(TaskError, "target population drift"):
        observe(fixture)

    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      validation_path = fixture / "validation.json"
      validation = json.loads(validation_path.read_text())
      validation["protocol_sha256"] = "0" * 64
      validation_path.write_text(json.dumps(validation))

      with self.assertRaisesRegex(TaskError, "protocol artifact drift"):
        observe(fixture)
if __name__ == "__main__":
  unittest.main()
