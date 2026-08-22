"""Reproduce the retained MBTA replay from checksum-pinned LAMP parquet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SERVICE_DATE = 20260818
ROUTE_ID = "Blue"
START_UTC = 1787050800
END_UTC = 1787058000
FEED_VERSION = "Summer 2026, 2026-08-17T19:35:03+00:00, version D"
TIMEOUT_SECONDS = 30
USER_AGENT = "tinymesh-research/0.1"


@dataclass(frozen=True)
class Source:
  name: str
  filename: str
  url: str
  bytes: int
  sha256: str


SOURCES = (
  Source(
    "performance",
    "2026-08-18-subway-on-time-performance-v1.parquet",
    "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/2026-08-18-subway-on-time-performance-v1.parquet",
    1_352_196,
    "ccd64aa2467191ca884f2a886a88e697764d7f74cd562e946345a3e0dd4a3e87",
  ),
  Source(
    "feed_info",
    "mbta-2026-feed_info.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/feed_info.parquet",
    172_361,
    "ddb5673f6f81b738071052231b7dd7aa4c58e8a6d34e33d2c76d290770fbb111",
  ),
  Source(
    "trips",
    "mbta-2026-trips.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/trips.parquet",
    13_689_076,
    "d6116af20771d213aa96f25bced2119b6fa7fa6e8393d3b0fae255cdf89574a5",
  ),
  Source(
    "stop_times",
    "mbta-2026-stop_times.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/stop_times.parquet",
    60_348_530,
    "842f13908c2b92ee565969ffd9ae341673c6d1e365b70a2a4c39f8d82d6ae630",
  ),
  Source(
    "stops",
    "mbta-2026-stops.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/stops.parquet",
    1_152_737,
    "3bce51fe3098cb17c898663b2023ee5f0381ca39d91a5bc1012767ad36e75658",
  ),
)


def acquire(directory: Path) -> None:
  """Download exact source bytes atomically, rejecting drift."""
  directory.mkdir(parents=True, exist_ok=True)
  for source in SOURCES:
    path = directory / source.filename
    if path.exists():
      _verify(path, source)
      continue
    temporary = path.with_suffix(path.suffix + ".part")
    request = Request(source.url, headers={"User-Agent": USER_AGENT})
    try:
      with urlopen(request, timeout=TIMEOUT_SECONDS) as response, temporary.open("wb") as output:
        remaining = source.bytes
        while chunk := response.read(min(1024 * 1024, remaining + 1)):
          remaining -= len(chunk)
          if remaining < 0:
            raise ValueError(f"{source.name}: exceeds declared {source.bytes} bytes")
          output.write(chunk)
      _verify(temporary, source)
      temporary.replace(path)
    except BaseException:
      temporary.unlink(missing_ok=True)
      raise


def extract(directory: Path) -> dict[str, Any]:
  """Return the deterministic projection from already acquired source bytes."""
  paths = {source.name: directory / source.filename for source in SOURCES}
  for source in SOURCES:
    _verify(paths[source.name], source)
  try:
    import duckdb
  except ImportError as error:
    raise RuntimeError("extraction requires duckdb==1.4.1") from error

  connection = duckdb.connect()
  _require_columns(
    connection,
    paths["performance"],
    {
      "service_date",
      "start_time",
      "trip_id",
      "vehicle_id",
      "route_id",
      "direction_id",
      "stop_id",
      "parent_station",
      "stop_sequence",
      "move_timestamp",
      "stop_timestamp",
      "travel_time_seconds",
      "dwell_time_seconds",
      "headway_trunk_seconds",
      "trunk_route_id",
      "scheduled_arrival_time",
      "scheduled_departure_time",
      "scheduled_travel_time",
      "scheduled_headway_trunk",
    },
  )
  for name, columns in {
    "feed_info": {"feed_version", "gtfs_active_date", "gtfs_end_date"},
    "trips": {"route_id", "service_id", "trip_id", "direction_id", "gtfs_active_date", "gtfs_end_date"},
    "stop_times": {"trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "gtfs_active_date", "gtfs_end_date"},
    "stops": {"stop_id", "stop_name", "parent_station", "stop_lat", "stop_lon", "gtfs_active_date", "gtfs_end_date"},
  }.items():
    _require_columns(connection, paths[name], columns)

  performance = str(paths["performance"])
  feed_info = str(paths["feed_info"])
  trips = str(paths["trips"])
  stop_times = str(paths["stop_times"])
  stops = str(paths["stops"])
  selection = (
    f"route_id = '{ROUTE_ID}' AND coalesce(move_timestamp, stop_timestamp) >= {START_UTC} "
    f"AND coalesce(move_timestamp, stop_timestamp) < {END_UTC}"
  )
  active = f"gtfs_active_date <= {SERVICE_DATE} AND gtfs_end_date >= {SERVICE_DATE}"
  active_trip = f"t.gtfs_active_date <= {SERVICE_DATE} AND t.gtfs_end_date >= {SERVICE_DATE}"
  active_call = f"s.gtfs_active_date <= {SERVICE_DATE} AND s.gtfs_end_date >= {SERVICE_DATE}"

  feed_rows = connection.execute(
    f"SELECT feed_version, gtfs_active_date, gtfs_end_date FROM read_parquet(?) WHERE {active}", [feed_info]
  ).fetchall()
  if feed_rows != [(FEED_VERSION, SERVICE_DATE, 20260819)]:
    raise ValueError(f"feed_info: unexpected active schedule {feed_rows!r}")

  replay_columns = (
    "service_date",
    "start_time",
    "trip_id",
    "vehicle_id",
    "route_id",
    "direction_id",
    "stop_id",
    "parent_station",
    "stop_sequence",
    "move_timestamp",
    "stop_timestamp",
    "travel_time_seconds",
    "dwell_time_seconds",
    "headway_trunk_seconds",
    "trunk_route_id",
    "scheduled_arrival_time",
    "scheduled_departure_time",
    "scheduled_travel_time",
    "scheduled_headway_trunk",
  )
  replay = _rows(
    connection,
    f"SELECT {', '.join(replay_columns)} FROM read_parquet(?) WHERE {selection} "
    "ORDER BY service_date, start_time, trip_id, stop_sequence",
    [performance],
    replay_columns,
  )
  if len(replay) != 663:
    raise ValueError(f"performance: expected 663 selected rows, got {len(replay)}")

  trip_columns = ("route_id", "service_id", "trip_id", "direction_id")
  selected_trips = f"SELECT DISTINCT trip_id FROM read_parquet(?) WHERE {selection}"
  trip_rows = _rows(
    connection,
    f"SELECT {', '.join('t.' + column for column in trip_columns)} FROM read_parquet(?) t "
    f"JOIN ({selected_trips}) r USING (trip_id) WHERE {active_trip} "
    "ORDER BY t.trip_id",
    [trips, performance],
    trip_columns,
  )
  if len(trip_rows) != 66:
    raise ValueError(f"trips: expected 66 selected rows, got {len(trip_rows)}")

  call_columns = ("trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time")
  calls = _rows(
    connection,
    f"SELECT {', '.join('s.' + column for column in call_columns)} FROM read_parquet(?) s "
    f"JOIN ({selected_trips}) r USING (trip_id) WHERE {active_call} "
    "ORDER BY s.trip_id, s.stop_sequence",
    [stop_times, performance],
    call_columns,
  )
  if len(calls) != 792:
    raise ValueError(f"stop_times: expected 792 selected rows, got {len(calls)}")

  stop_columns = ("stop_id", "stop_name", "parent_station", "stop_lat", "stop_lon")
  selected_stops = (
    f"SELECT DISTINCT s.stop_id FROM read_parquet(?) s JOIN ({selected_trips}) r USING (trip_id) "
    f"WHERE {active_call}"
  )
  stop_rows = _rows(
    connection,
    f"SELECT {', '.join('s.' + column for column in stop_columns)} FROM read_parquet(?) s "
    f"JOIN ({selected_stops}) r USING (stop_id) WHERE {active_call} ORDER BY s.stop_id",
    [stops, stop_times, performance],
    stop_columns,
  )
  if len(stop_rows) != 24:
    raise ValueError(f"stops: expected 24 selected rows, got {len(stop_rows)}")

  full_day = connection.execute(
    "SELECT count(*), count(DISTINCT (service_date, start_time, trip_id)), count(DISTINCT vehicle_id), "
    "count(*) FILTER (WHERE move_timestamp IS NULL), count(*) FILTER (WHERE stop_timestamp IS NULL) "
    "FROM read_parquet(?)",
    [performance],
  ).fetchone()
  duplicate_groups = connection.execute(
    "SELECT count(*) FROM (SELECT service_date, start_time, trip_id, stop_sequence, count(*) n "
    "FROM read_parquet(?) GROUP BY ALL HAVING n > 1)",
    [performance],
  ).fetchone()[0]
  headway = connection.execute(
    "WITH departures AS ("
    "SELECT *, lead(move_timestamp) OVER ("
    "PARTITION BY service_date, start_time, trip_id, vehicle_id "
    "ORDER BY coalesce(move_timestamp, stop_timestamp), stop_sequence"
    ") target_time FROM read_parquet(?) WHERE route_id = ?"
    "), derived AS ("
    "SELECT *, target_time - lag(target_time) OVER ("
    "PARTITION BY parent_station, trunk_route_id, direction_id ORDER BY target_time"
    ") derived_headway FROM departures"
    ") SELECT count(*), count(headway_trunk_seconds), count(derived_headway), "
    "count(*) FILTER (WHERE headway_trunk_seconds = derived_headway), "
    "count(*) FILTER (WHERE headway_trunk_seconds IS NOT NULL AND derived_headway IS NOT NULL "
    "AND headway_trunk_seconds <> derived_headway) FROM derived",
    [performance, ROUTE_ID],
  ).fetchone()

  return {
    "schema": 2,
    "source": {
      "publisher": "Massachusetts Bay Transportation Authority via LAMP",
      "lamp_revision": "e266440db994ed33eede5e44a137b205e4a1e8dd",
      "gtfs_documentation_revision": "02da961b963ba3d3a66042ca4d5bd19e21ce5c0a",
      "license_url": "https://cdn.mbta.com/sites/default/files/developers/2018-10-30-massdot-developers-license-agreement.pdf",
      "license_sha256": "e791e24e86b9e974de060c3b238abfc392260cad54ccc0d2d3d4f0a61846b91a",
      "files": [asdict(source) for source in SOURCES],
    },
    "extraction": {
      "tool": "duckdb==1.4.1",
      "service_date": SERVICE_DATE,
      "route_id": ROUTE_ID,
      "interval_utc": [START_UTC, END_UTC],
      "interval_local": ["2026-08-18T07:00:00-04:00", "2026-08-18T09:00:00-04:00"],
      "timezone": "America/New_York",
      "event_selector": "coalesce(move_timestamp, stop_timestamp)",
    },
    "schedule": {
      "feed_version": FEED_VERSION,
      "active_dates": [SERVICE_DATE, 20260819],
      "trips": trip_rows,
      "calls": calls,
      "stops": stop_rows,
    },
    "replay": replay,
    "source_audit": {
      "full_day_rows": full_day[0],
      "full_day_trip_instances": full_day[1],
      "full_day_vehicles": full_day[2],
      "full_day_missing_move": full_day[3],
      "full_day_missing_stop": full_day[4],
      "full_day_duplicate_trip_stops": duplicate_groups,
      "blue_rows": headway[0],
      "blue_source_headways": headway[1],
      "blue_derived_headways": headway[2],
      "blue_exact_headways": headway[3],
      "blue_headway_mismatches": headway[4],
    },
    "provenance": {
      "move_timestamp": "observed_vehicle_position",
      "stop_timestamp": "mixed_vehicle_position_or_trip_update_prediction",
      "travel_time_seconds": "derived_mixed_stop_minus_observed_move",
      "dwell_time_seconds": "derived_observed_next_move_minus_mixed_stop",
      "headway_trunk_seconds": "derived_successive_observed_next_moves",
      "scheduled_arrival_time": "schedule",
      "scheduled_departure_time": "schedule",
      "scheduled_travel_time": "schedule_derived",
      "scheduled_headway_trunk": "schedule_derived",
    },
  }


def _verify(path: Path, source: Source) -> None:
  if not path.is_file():
    raise FileNotFoundError(f"{source.name}: missing {path}")
  size = path.stat().st_size
  if size != source.bytes:
    raise ValueError(f"{source.name}: expected {source.bytes} bytes, got {size}")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if digest != source.sha256:
    raise ValueError(f"{source.name}: expected sha256 {source.sha256}, got {digest}")


def _require_columns(connection: Any, path: Path, required: set[str]) -> None:
  columns = {row[0] for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
  missing = required - columns
  if missing:
    raise ValueError(f"{path.name}: missing columns {', '.join(sorted(missing))}")


def _rows(connection: Any, query: str, parameters: list[str], columns: tuple[str, ...]) -> list[dict[str, Any]]:
  return [dict(zip(columns, row)) for row in connection.execute(query, parameters).fetchall()]


def write(directory: Path, value: dict[str, Any]) -> None:
  """Write one manifest and four transparent tables without partial files."""
  directory.mkdir(parents=True, exist_ok=True)
  schedule = value["schedule"]
  tables = {
    "replay.csv": value["replay"],
    "schedule_trips.csv": schedule["trips"],
    "schedule_calls.csv": schedule["calls"],
    "schedule_stops.csv": schedule["stops"],
  }
  artifacts = {}
  for filename, rows in tables.items():
    path = directory / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
      writer = csv.DictWriter(output, fieldnames=rows[0], lineterminator="\n")
      writer.writeheader()
      writer.writerows(rows)
    temporary.replace(path)
    artifacts[filename] = {
      "rows": len(rows),
      "bytes": path.stat().st_size,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

  manifest = {key: item for key, item in value.items() if key not in {"replay", "schedule"}}
  manifest["schedule"] = {
    "feed_version": schedule["feed_version"],
    "active_dates": schedule["active_dates"],
  }
  manifest["artifacts"] = artifacts
  encoded = json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  path = directory / "manifest.json"
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--download", action="store_true")
  arguments = parser.parse_args()
  if arguments.download:
    acquire(arguments.source_dir)
  write(arguments.output_dir, extract(arguments.source_dir))


if __name__ == "__main__":
  main()
