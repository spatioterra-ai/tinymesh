import csv
import hashlib
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
        result.missing.move_timestamp,
        result.missing.stop_timestamp,
        result.missing.travel_time,
        result.missing.dwell_time,
        result.missing.trunk_headway,
        result.missing.scheduled_arrival,
        result.missing.scheduled_departure,
        result.missing.scheduled_travel_time,
        result.missing.scheduled_trunk_headway,
        result.missing.vehicle_id,
        result.missing.stop_id,
      ),
      (0, 0, 0, 110, 55, 0, 0, 55, 0, 0, 0),
    )
    self.assertEqual(result.missing.observation_as_of, result.source_rows)

  def test_target_lineage_is_field_specific(self) -> None:
    result = observe()

    self.assertEqual(result.lineage.observed_vehicle_positions, 663)
    self.assertEqual(result.lineage.mixed_stop_timestamps, 663)
    self.assertEqual(result.lineage.mixed_travel_times, 663)
    self.assertEqual(result.lineage.mixed_dwell_times, 553)
    self.assertEqual(result.lineage.observed_trunk_headways, 608)
    self.assertFalse(result.observation_age_available)
    self.assertEqual(result.arrival_target_decision, "reject:mixed_stop_lineage")
    self.assertEqual(result.travel_time_target_decision, "reject:mixed_stop_lineage")
    self.assertEqual(result.dwell_target_decision, "reject:mixed_stop_lineage")
    self.assertEqual(result.headway_target_decision, "extend_replay:observed_movement_headway")

  def test_observed_headway_reproduces_exactly(self) -> None:
    headway = observe().headway

    self.assertEqual(
      (
        headway.source_values,
        headway.derived_values,
        headway.exact_matches,
        headway.mismatches,
        headway.boundary_only,
        headway.derived_only,
      ),
      (608, 575, 575, 0, 33, 0),
    )
    self.assertEqual((headway.physical_departures, headway.duplicate_aliases, headway.conflicting_aliases), (597, 0, 0))
    self.assertEqual((headway.parent_stations, headway.station_directions), (12, 22))
    self.assertEqual((headway.target_bins, headway.colliding_target_bins, headway.max_targets_per_bin), (485, 90, 2))
    result = observe()
    self.assertEqual((result.schedule_union_edges, result.active_edges), (22, 22))

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

  def test_rejects_headway_that_does_not_match_movement_events(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      replay_path = fixture / "replay.csv"
      with replay_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
      for row in rows:
        if row["headway_trunk_seconds"]:
          row["headway_trunk_seconds"] = str(int(row["headway_trunk_seconds"]) + 1)
      with replay_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
      manifest_path = fixture / "manifest.json"
      manifest = json.loads(manifest_path.read_text())
      data = replay_path.read_bytes()
      manifest["artifacts"]["replay.csv"]["bytes"] = len(data)
      manifest["artifacts"]["replay.csv"]["sha256"] = hashlib.sha256(data).hexdigest()
      manifest_path.write_text(json.dumps(manifest))

      with self.assertRaisesRegex(ReplayError, "headway: source mismatch"):
        observe(fixture)


if __name__ == "__main__":
  unittest.main()
