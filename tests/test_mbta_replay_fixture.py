import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.tools.mbta_replay_extract import END_UTC, ROUTE_ID, SERVICE_DATE, START_UTC, Source, _verify


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "fixtures" / "mbta_replay"


def rows(filename: str) -> list[dict[str, str]]:
  with (FIXTURE / filename).open(newline="") as source:
    return list(csv.DictReader(source))


def seconds(value: str) -> int:
  hour, minute, second = map(int, value.split(":"))
  return hour * 3600 + minute * 60 + second


class MbtaReplayFixtureTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.manifest = json.loads((FIXTURE / "manifest.json").read_text())
    cls.replay = rows("replay.csv")
    cls.trips = rows("schedule_trips.csv")
    cls.calls = rows("schedule_calls.csv")
    cls.stops = rows("schedule_stops.csv")

  def test_manifest_pins_sources_schedule_and_derived_tables(self) -> None:
    files = {item["name"]: item for item in self.manifest["source"]["files"]}

    self.assertEqual(set(files), {"performance", "feed_info", "trips", "stop_times", "stops"})
    self.assertTrue(all(item["bytes"] > 0 and len(item["sha256"]) == 64 for item in files.values()))
    self.assertEqual(self.manifest["schedule"]["active_dates"], [SERVICE_DATE, 20260819])
    self.assertEqual(
      self.manifest["schedule"]["feed_version"],
      "Summer 2026, 2026-08-17T19:35:03+00:00, version D",
    )
    self.assertEqual(self.manifest["extraction"]["interval_utc"], [START_UTC, END_UTC])
    self.assertEqual(self.manifest["extraction"]["timezone"], "America/New_York")
    for filename, artifact in self.manifest["artifacts"].items():
      path = FIXTURE / filename
      self.assertEqual(path.stat().st_size, artifact["bytes"])
      self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])

  def test_replay_is_exact_bounded_and_identity_ordered(self) -> None:
    identities = [(row["service_date"], int(row["start_time"]), row["trip_id"], int(row["stop_sequence"])) for row in self.replay]

    self.assertEqual(len(self.replay), 663)
    self.assertEqual(len(set(identities)), 663)
    self.assertEqual(identities, sorted(identities))
    self.assertEqual({row["route_id"] for row in self.replay}, {ROUTE_ID})
    self.assertEqual({int(row["service_date"]) for row in self.replay}, {SERVICE_DATE})
    self.assertTrue(all(START_UTC <= int(row["move_timestamp"] or row["stop_timestamp"]) < END_UTC for row in self.replay))
    self.assertEqual(len({(row["service_date"], row["start_time"], row["trip_id"]) for row in self.replay}), 66)
    self.assertEqual(len({row["vehicle_id"] for row in self.replay}), 12)

  def test_every_replay_row_resolves_to_the_pinned_schedule(self) -> None:
    trips = {row["trip_id"]: row for row in self.trips}
    calls = {(row["trip_id"], row["stop_sequence"]): row for row in self.calls}

    self.assertEqual((len(trips), len(calls), len(self.stops)), (66, 792, 24))
    for row in self.replay:
      trip = trips[row["trip_id"]]
      call = calls[(row["trip_id"], row["stop_sequence"])]
      direction = str(int(row["direction_id"] == "True"))
      self.assertEqual((trip["route_id"], trip["direction_id"]), (row["route_id"], direction))
      self.assertEqual(call["stop_id"], row["stop_id"])
      self.assertEqual(seconds(call["arrival_time"]), int(row["scheduled_arrival_time"]))
      self.assertEqual(seconds(call["departure_time"]), int(row["scheduled_departure_time"]))

  def test_timestamp_lineage_does_not_invent_observed_arrivals(self) -> None:
    self.assertEqual(self.manifest["provenance"]["move_timestamp"], "observed_vehicle_position")
    self.assertEqual(
      self.manifest["provenance"]["stop_timestamp"],
      "mixed_vehicle_position_or_trip_update_prediction",
    )
    self.assertNotIn("observed_arrival", self.manifest["provenance"].values())
    self.assertEqual(
      self.manifest["provenance"],
      {
        "dwell_time_seconds": "derived_observed_next_move_minus_mixed_stop",
        "headway_trunk_seconds": "derived_successive_observed_next_moves",
        "move_timestamp": "observed_vehicle_position",
        "scheduled_arrival_time": "schedule",
        "scheduled_departure_time": "schedule",
        "scheduled_headway_trunk": "schedule_derived",
        "scheduled_travel_time": "schedule_derived",
        "stop_timestamp": "mixed_vehicle_position_or_trip_update_prediction",
        "travel_time_seconds": "derived_mixed_stop_minus_observed_move",
      },
    )
    self.assertEqual(
      self.manifest["source_audit"],
      {
        "blue_derived_headways": 4274,
        "blue_exact_headways": 4274,
        "blue_headway_mismatches": 0,
        "blue_rows": 4706,
        "blue_source_headways": 4274,
        "full_day_duplicate_trip_stops": 59,
        "full_day_missing_move": 2242,
        "full_day_missing_stop": 583,
        "full_day_rows": 41446,
        "full_day_trip_instances": 3722,
        "full_day_vehicles": 241,
      },
    )

  def test_fixture_retains_only_fields_needed_for_target_audit(self) -> None:
    self.assertEqual(sum(bool(row["headway_trunk_seconds"]) for row in self.replay), 608)
    self.assertEqual(sum(bool(row["travel_time_seconds"]) for row in self.replay), 663)
    self.assertEqual(sum(bool(row["dwell_time_seconds"]) for row in self.replay), 553)
    self.assertEqual(sum(bool(row["scheduled_headway_trunk"]) for row in self.replay), 663)
    self.assertEqual({row["trunk_route_id"] for row in self.replay}, {"Blue"})
    self.assertEqual(len({row["parent_station"] for row in self.replay}), 12)

  def test_source_guard_rejects_size_and_checksum_drift(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "source.bin"
      path.write_bytes(b"exact")
      source = Source("test", path.name, "https://example.invalid/source", 5, hashlib.sha256(b"exact").hexdigest())
      _verify(path, source)
      with self.assertRaisesRegex(ValueError, "expected 4 bytes"):
        _verify(path, Source("test", path.name, source.url, 4, source.sha256))
      with self.assertRaisesRegex(ValueError, "expected sha256"):
        _verify(path, Source("test", path.name, source.url, 5, "0" * 64))

  def test_duckdb_is_an_extraction_tool_not_a_project_dependency(self) -> None:
    self.assertNotIn("duckdb", (ROOT / "pyproject.toml").read_text())


if __name__ == "__main__":
  unittest.main()
