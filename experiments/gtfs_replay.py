"""Audit the retained version-aligned MBTA replay without inventing labels."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "mbta_replay"
BIN_SECONDS = 300
REPLAY_FIELDS = (
  "service_date",
  "start_time",
  "trip_id",
  "vehicle_id",
  "route_id",
  "direction_id",
  "stop_id",
  "stop_sequence",
  "move_timestamp",
  "stop_timestamp",
  "scheduled_arrival_time",
  "scheduled_departure_time",
)


class ReplayError(ValueError):
  """A retained replay violates its manifest or semantic contract."""


@dataclass(frozen=True, order=True)
class TripInstance:
  service_date: int
  start_time: int
  trip_id: str


@dataclass(frozen=True)
class ReplayRow:
  instance: TripInstance
  vehicle_id: str
  route_id: str
  direction_id: int
  stop_id: str
  stop_sequence: int
  move_timestamp: int | None
  stop_timestamp: int | None
  scheduled_arrival_time: int | None
  scheduled_departure_time: int | None


@dataclass(frozen=True)
class IntervalAudit:
  start_utc: int
  rows: int
  trip_instances: int
  vehicles: int
  active_edges: int
  colliding_stops: int


@dataclass(frozen=True)
class Observation:
  source_rows: int
  trip_instances: int
  vehicles: int
  stops: int
  schedule_rows_resolved: int
  duplicate_trip_stops: int
  missing_move_timestamp: int
  missing_stop_timestamp: int
  missing_scheduled_arrival: int
  missing_scheduled_departure: int
  missing_vehicle_id: int
  missing_stop_id: int
  missing_observation_as_of: int
  observation_age_available: bool
  observed_vehicle_positions: int
  mixed_stop_timestamps: int
  observed_arrival_targets: int
  schedule_union_edges: int
  active_edges: int
  colliding_stop_bins: int
  max_vehicles_per_stop_bin: int
  intervals: tuple[IntervalAudit, ...]
  stage_3_decision: str


def observe(path: str | Path = FIXTURE) -> Observation:
  """Validate and audit the retained replay using only standard-library code."""
  directory = Path(path)
  manifest = json.loads((directory / "manifest.json").read_text())
  tables = {filename: _table(directory, filename, artifact) for filename, artifact in manifest["artifacts"].items()}
  replay = tuple(_replay(row, number) for number, row in enumerate(tables["replay.csv"], start=2))
  _validate_manifest(manifest)
  _validate_replay(replay, manifest)

  trips = {row["trip_id"]: row for row in tables["schedule_trips.csv"]}
  calls_by_trip = _calls(tables["schedule_calls.csv"])
  schedule_stops = {row["stop_id"] for row in tables["schedule_stops.csv"]}
  incoming, union_edges = _topology(calls_by_trip)
  resolved = _resolve(replay, trips, calls_by_trip, schedule_stops)

  intervals: dict[int, list[ReplayRow]] = defaultdict(list)
  stop_vehicles: dict[tuple[int, str], set[str]] = defaultdict(set)
  interval_edges: dict[int, set[tuple[str, str]]] = defaultdict(set)
  for row in replay:
    event_time = row.move_timestamp if row.move_timestamp is not None else row.stop_timestamp
    if event_time is None:
      continue
    interval = event_time // BIN_SECONDS * BIN_SECONDS
    intervals[interval].append(row)
    stop_vehicles[(interval, row.stop_id)].add(row.vehicle_id)
    edge = incoming.get((row.instance.trip_id, row.stop_sequence))
    if edge is not None:
      interval_edges[interval].add(edge)

  interval_audits = tuple(
    IntervalAudit(
      interval,
      len(rows),
      len({row.instance for row in rows}),
      len({row.vehicle_id for row in rows}),
      len(interval_edges[interval]),
      sum(len(vehicles) > 1 for (start, _), vehicles in stop_vehicles.items() if start == interval),
    )
    for interval, rows in sorted(intervals.items())
  )
  collisions = [len(vehicles) for vehicles in stop_vehicles.values() if len(vehicles) > 1]
  active_edges = set().union(*interval_edges.values())
  provenance = manifest["provenance"]
  return Observation(
    len(replay),
    len({row.instance for row in replay}),
    len({row.vehicle_id for row in replay}),
    len({row.stop_id for row in replay}),
    resolved,
    len(replay) - len({(row.instance, row.stop_sequence) for row in replay}),
    sum(row.move_timestamp is None for row in replay),
    sum(row.stop_timestamp is None for row in replay),
    sum(row.scheduled_arrival_time is None for row in replay),
    sum(row.scheduled_departure_time is None for row in replay),
    sum(not row.vehicle_id for row in replay),
    sum(not row.stop_id for row in replay),
    len(replay),
    False,
    sum(row.move_timestamp is not None and provenance["move_timestamp"] == "observed_vehicle_position" for row in replay),
    sum(row.stop_timestamp is not None and provenance["stop_timestamp"] == "mixed_vehicle_position_or_trip_update_prediction" for row in replay),
    0,
    len(union_edges),
    len(active_edges),
    len(collisions),
    max(collisions, default=1),
    interval_audits,
    "blocked:no_source_tagged_observed_arrival_target",
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _table(directory: Path, filename: str, artifact: dict) -> list[dict[str, str]]:
  path = directory / filename
  if not path.is_file():
    raise ReplayError(f"{filename}: missing retained artifact")
  data = path.read_bytes()
  digest = hashlib.sha256(data).hexdigest()
  if (len(data), digest) != (artifact["bytes"], artifact["sha256"]):
    raise ReplayError(f"{filename}: does not match retained artifact manifest")
  with path.open(newline="") as source:
    rows = list(csv.DictReader(source))
  if len(rows) != artifact["rows"]:
    raise ReplayError(f"{filename}: expected {artifact['rows']} rows, got {len(rows)}")
  return rows


def _replay(row: dict[str, str], line: int) -> ReplayRow:
  if tuple(row) != REPLAY_FIELDS:
    raise ReplayError(f"replay.csv:{line}: unexpected fields")
  try:
    direction = {"False": 0, "True": 1}[row["direction_id"]]
    return ReplayRow(
      TripInstance(int(row["service_date"]), int(row["start_time"]), row["trip_id"]),
      row["vehicle_id"],
      row["route_id"],
      direction,
      row["stop_id"],
      int(row["stop_sequence"]),
      _optional_int(row["move_timestamp"]),
      _optional_int(row["stop_timestamp"]),
      _optional_int(row["scheduled_arrival_time"]),
      _optional_int(row["scheduled_departure_time"]),
    )
  except (KeyError, ValueError) as error:
    raise ReplayError(f"replay.csv:{line}: invalid typed value") from error


def _validate_manifest(manifest: dict) -> None:
  if manifest["extraction"]["event_selector"] != "coalesce(move_timestamp, stop_timestamp)":
    raise ReplayError("manifest: unsupported event selector")
  if manifest["provenance"] != {
    "move_timestamp": "observed_vehicle_position",
    "scheduled_arrival_time": "schedule",
    "scheduled_departure_time": "schedule",
    "stop_timestamp": "mixed_vehicle_position_or_trip_update_prediction",
  }:
    raise ReplayError("manifest: unsupported provenance contract")


def _validate_replay(replay: tuple[ReplayRow, ...], manifest: dict) -> None:
  start, end = manifest["extraction"]["interval_utc"]
  identities = [(row.instance, row.stop_sequence) for row in replay]
  if identities != sorted(identities):
    raise ReplayError("replay.csv: rows are not in canonical identity order")
  if len(identities) != len(set(identities)):
    raise ReplayError("replay.csv: duplicate trip-stop identity")
  for row in replay:
    event_time = row.move_timestamp if row.move_timestamp is not None else row.stop_timestamp
    if row.route_id != manifest["extraction"]["route_id"] or row.instance.service_date != manifest["extraction"]["service_date"]:
      raise ReplayError(f"replay.csv: row outside declared route or service date: {row.instance!r}")
    if event_time is None or not start <= event_time < end:
      raise ReplayError(f"replay.csv: row outside declared interval: {row.instance!r}")


def _calls(rows: list[dict[str, str]]) -> dict[str, tuple[dict[str, str], ...]]:
  grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    grouped[row["trip_id"]].append(row)
  return {trip_id: tuple(sorted(calls, key=lambda row: int(row["stop_sequence"]))) for trip_id, calls in grouped.items()}


def _topology(calls_by_trip: dict[str, tuple[dict[str, str], ...]]) -> tuple[dict[tuple[str, int], tuple[str, str]], set[tuple[str, str]]]:
  incoming = {}
  edges = set()
  for trip_id, calls in calls_by_trip.items():
    for previous, current in zip(calls, calls[1:]):
      edge = (previous["stop_id"], current["stop_id"])
      incoming[(trip_id, int(current["stop_sequence"]))] = edge
      edges.add(edge)
  return incoming, edges


def _resolve(
  replay: tuple[ReplayRow, ...],
  trips: dict[str, dict[str, str]],
  calls_by_trip: dict[str, tuple[dict[str, str], ...]],
  stops: set[str],
) -> int:
  calls = {(trip_id, int(call["stop_sequence"])): call for trip_id, trip_calls in calls_by_trip.items() for call in trip_calls}
  for row in replay:
    trip = trips.get(row.instance.trip_id)
    call = calls.get((row.instance.trip_id, row.stop_sequence))
    if trip is None or call is None or row.stop_id not in stops:
      raise ReplayError(f"schedule: unresolved identity {row.instance!r} stop_sequence={row.stop_sequence}")
    if (trip["route_id"], int(trip["direction_id"])) != (row.route_id, row.direction_id) or call["stop_id"] != row.stop_id:
      raise ReplayError(f"schedule: mismatched identity {row.instance!r} stop_sequence={row.stop_sequence}")
    if (_time(call["arrival_time"]), _time(call["departure_time"])) != (
      row.scheduled_arrival_time,
      row.scheduled_departure_time,
    ):
      raise ReplayError(f"schedule: mismatched time {row.instance!r} stop_sequence={row.stop_sequence}")
  return len(replay)


def _time(value: str) -> int:
  hour, minute, second = map(int, value.split(":"))
  return hour * 3600 + minute * 60 + second


def _optional_int(value: str) -> int | None:
  return int(value) if value else None


if __name__ == "__main__":
  main()
