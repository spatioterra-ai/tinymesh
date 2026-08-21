import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.gtfs_replay import FIXTURE, ReplayError, observe


class GtfsReplayTest(unittest.TestCase):
  def test_exact_audit(self) -> None:
    result = observe()

    self.assertEqual(
      (
        result.source_rows,
        result.trip_instances,
        result.vehicles,
        result.stops,
        result.schedule_rows_resolved,
        result.duplicate_trip_stops,
      ),
      (663, 66, 12, 24, 663, 0),
    )
    self.assertEqual(
      (
        result.missing_move_timestamp,
        result.missing_stop_timestamp,
        result.missing_scheduled_arrival,
        result.missing_scheduled_departure,
        result.missing_vehicle_id,
        result.missing_stop_id,
      ),
      (0, 0, 0, 0, 0, 0),
    )
    self.assertEqual(result.missing_observation_as_of, result.source_rows)

  def test_interval_and_topology_activity_is_explicit(self) -> None:
    result = observe()

    self.assertEqual((result.schedule_union_edges, result.active_edges), (22, 22))
    self.assertEqual((result.colliding_stop_bins, result.max_vehicles_per_stop_bin), (106, 2))
    self.assertEqual(len(result.intervals), 24)
    self.assertEqual([interval.start_utc for interval in result.intervals], list(range(1787050800, 1787058000, 300)))
    self.assertEqual((min(item.rows for item in result.intervals), max(item.rows for item in result.intervals)), (25, 30))
    self.assertEqual((min(item.active_edges for item in result.intervals), max(item.active_edges for item in result.intervals)), (21, 22))
    self.assertEqual(sum(item.rows for item in result.intervals), result.source_rows)

  def test_mixed_lineage_blocks_the_forecast_target(self) -> None:
    result = observe()

    self.assertEqual(result.observed_vehicle_positions, 663)
    self.assertEqual(result.mixed_stop_timestamps, 663)
    self.assertEqual(result.observed_arrival_targets, 0)
    self.assertFalse(result.observation_age_available)
    self.assertEqual(result.stage_3_decision, "blocked:no_source_tagged_observed_arrival_target")

  def test_repeated_audit_is_deterministic(self) -> None:
    self.assertEqual(observe(), observe())

  def test_rejects_artifact_drift_before_audit(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      replay = fixture / "replay.csv"
      replay.write_bytes(replay.read_bytes() + b"\n")

      with self.assertRaisesRegex(ReplayError, "does not match retained artifact manifest"):
        observe(fixture)

  def test_rejects_provenance_relabeling(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      manifest_path = fixture / "manifest.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["provenance"]["stop_timestamp"] = "observed_arrival"
      manifest_path.write_text(json.dumps(manifest))

      with self.assertRaisesRegex(ReplayError, "unsupported provenance contract"):
        observe(fixture)


if __name__ == "__main__":
  unittest.main()
