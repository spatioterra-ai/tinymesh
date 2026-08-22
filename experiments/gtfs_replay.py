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
  parent_station: str
  stop_sequence: int
  move_timestamp: int | None
  stop_timestamp: int | None
  travel_time_seconds: int | None
  dwell_time_seconds: int | None
  headway_trunk_seconds: int | None
  trunk_route_id: str
  scheduled_arrival_time: int | None
  scheduled_departure_time: int | None
  scheduled_travel_time: int | None
  scheduled_headway_trunk: int | None


@dataclass(frozen=True, order=True)
class ScheduleCall:
  trip_id: str
  stop_sequence: int
  stop_id: str
  arrival_time: int
  departure_time: int


@dataclass(frozen=True)
class RetainedReplay:
  rows: tuple[ReplayRow, ...]
  calls: tuple[ScheduleCall, ...]
  interval_utc: tuple[int, int]


@dataclass(frozen=True)
class Missingness:
  move_timestamp: int
  stop_timestamp: int
  travel_time: int
  dwell_time: int
  trunk_headway: int
  scheduled_arrival: int
  scheduled_departure: int
  scheduled_travel_time: int
  scheduled_trunk_headway: int
  vehicle_id: int
  stop_id: int
  observation_as_of: int


@dataclass(frozen=True)
class LineageAudit:
  observed_vehicle_positions: int
  mixed_stop_timestamps: int
  mixed_travel_times: int
  mixed_dwell_times: int
  observed_trunk_headways: int


@dataclass(frozen=True)
class HeadwayAudit:
  source_values: int
  derived_values: int
  exact_matches: int
  mismatches: int
  boundary_only: int
  derived_only: int
  physical_departures: int
  duplicate_aliases: int
  conflicting_aliases: int
  parent_stations: int
  station_directions: int
  target_bins: int
  colliding_target_bins: int
  max_targets_per_bin: int


@dataclass(frozen=True)
class Observation:
  source_rows: int
  trip_instances: int
  vehicles: int
  stops: int
  schedule_rows_resolved: int
  duplicate_trip_stops: int
  observation_age_available: bool
  missing: Missingness
  lineage: LineageAudit
  headway: HeadwayAudit
  schedule_union_edges: int
  active_edges: int
  arrival_target_decision: str
  travel_time_target_decision: str
  dwell_target_decision: str
  headway_target_decision: str


def observe(path: str | Path = FIXTURE) -> Observation:
  """Validate and audit the retained replay using only standard-library code."""
  source = load(path)
  replay = source.rows
  calls_by_trip = _calls_by_trip(source.calls)

  directory = Path(path)
  manifest = json.loads((directory / "manifest.json").read_text())
  incoming, union_edges = _topology(calls_by_trip)
  headway = _headways(replay, calls_by_trip)

  active_edges = {
    edge
    for row in replay
    if (edge := incoming.get((row.instance.trip_id, row.stop_sequence))) is not None
  }
  provenance = manifest["provenance"]
  return Observation(
    source_rows=len(replay),
    trip_instances=len({row.instance for row in replay}),
    vehicles=len({row.vehicle_id for row in replay}),
    stops=len({row.stop_id for row in replay}),
    schedule_rows_resolved=len(replay),
    duplicate_trip_stops=len(replay) - len({(row.instance, row.stop_sequence) for row in replay}),
    observation_age_available=False,
    missing=Missingness(
      move_timestamp=sum(row.move_timestamp is None for row in replay),
      stop_timestamp=sum(row.stop_timestamp is None for row in replay),
      travel_time=sum(row.travel_time_seconds is None for row in replay),
      dwell_time=sum(row.dwell_time_seconds is None for row in replay),
      trunk_headway=sum(row.headway_trunk_seconds is None for row in replay),
      scheduled_arrival=sum(row.scheduled_arrival_time is None for row in replay),
      scheduled_departure=sum(row.scheduled_departure_time is None for row in replay),
      scheduled_travel_time=sum(row.scheduled_travel_time is None for row in replay),
      scheduled_trunk_headway=sum(row.scheduled_headway_trunk is None for row in replay),
      vehicle_id=sum(not row.vehicle_id for row in replay),
      stop_id=sum(not row.stop_id for row in replay),
      observation_as_of=len(replay),
    ),
    lineage=LineageAudit(
      observed_vehicle_positions=sum(
        row.move_timestamp is not None and provenance["move_timestamp"] == "observed_vehicle_position" for row in replay
      ),
      mixed_stop_timestamps=sum(
        row.stop_timestamp is not None and provenance["stop_timestamp"] == "mixed_vehicle_position_or_trip_update_prediction"
        for row in replay
      ),
      mixed_travel_times=sum(
        row.travel_time_seconds is not None and provenance["travel_time_seconds"] == "derived_mixed_stop_minus_observed_move"
        for row in replay
      ),
      mixed_dwell_times=sum(
        row.dwell_time_seconds is not None and provenance["dwell_time_seconds"] == "derived_observed_next_move_minus_mixed_stop"
        for row in replay
      ),
      observed_trunk_headways=sum(
        row.headway_trunk_seconds is not None
        and provenance["headway_trunk_seconds"] == "derived_successive_observed_next_moves"
        for row in replay
      ),
    ),
    headway=headway,
    schedule_union_edges=len(union_edges),
    active_edges=len(active_edges),
    arrival_target_decision="reject:mixed_stop_lineage",
    travel_time_target_decision="reject:mixed_stop_lineage",
    dwell_target_decision="reject:mixed_stop_lineage",
    headway_target_decision="extend_replay:observed_movement_headway",
  )


def load(path: str | Path = FIXTURE) -> RetainedReplay:
  """Load one checksum-pinned replay after validating its source contract."""
  directory = Path(path)
  manifest = json.loads((directory / "manifest.json").read_text())
  tables = {filename: _table(directory, filename, artifact) for filename, artifact in manifest["artifacts"].items()}
  replay = tuple(_replay(row, number) for number, row in enumerate(tables["replay.csv"], start=2))
  _validate_manifest(manifest)
  _validate_replay(replay, manifest)

  trips = {row["trip_id"]: row for row in tables["schedule_trips.csv"]}
  calls = tuple(_schedule_call(row) for row in tables["schedule_calls.csv"])
  calls_by_trip = _calls_by_trip(calls)
  schedule_stops = {row["stop_id"] for row in tables["schedule_stops.csv"]}
  _resolve(replay, trips, calls_by_trip, schedule_stops)
  return RetainedReplay(replay, calls, tuple(manifest["extraction"]["interval_utc"]))


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
      instance=TripInstance(int(row["service_date"]), int(row["start_time"]), row["trip_id"]),
      vehicle_id=row["vehicle_id"],
      route_id=row["route_id"],
      direction_id=direction,
      stop_id=row["stop_id"],
      parent_station=row["parent_station"],
      stop_sequence=int(row["stop_sequence"]),
      move_timestamp=_optional_int(row["move_timestamp"]),
      stop_timestamp=_optional_int(row["stop_timestamp"]),
      travel_time_seconds=_optional_int(row["travel_time_seconds"]),
      dwell_time_seconds=_optional_int(row["dwell_time_seconds"]),
      headway_trunk_seconds=_optional_int(row["headway_trunk_seconds"]),
      trunk_route_id=row["trunk_route_id"],
      scheduled_arrival_time=_optional_int(row["scheduled_arrival_time"]),
      scheduled_departure_time=_optional_int(row["scheduled_departure_time"]),
      scheduled_travel_time=_optional_int(row["scheduled_travel_time"]),
      scheduled_headway_trunk=_optional_int(row["scheduled_headway_trunk"]),
    )
  except (KeyError, ValueError) as error:
    raise ReplayError(f"replay.csv:{line}: invalid typed value") from error


def _validate_manifest(manifest: dict) -> None:
  if manifest["schema"] != 2:
    raise ReplayError("manifest: unsupported schema")
  if manifest["extraction"]["event_selector"] != "coalesce(move_timestamp, stop_timestamp)":
    raise ReplayError("manifest: unsupported event selector")
  if manifest["provenance"] != {
    "dwell_time_seconds": "derived_observed_next_move_minus_mixed_stop",
    "headway_trunk_seconds": "derived_successive_observed_next_moves",
    "move_timestamp": "observed_vehicle_position",
    "scheduled_arrival_time": "schedule",
    "scheduled_departure_time": "schedule",
    "scheduled_headway_trunk": "schedule_derived",
    "scheduled_travel_time": "schedule_derived",
    "stop_timestamp": "mixed_vehicle_position_or_trip_update_prediction",
    "travel_time_seconds": "derived_mixed_stop_minus_observed_move",
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


def _schedule_call(row: dict[str, str]) -> ScheduleCall:
  return ScheduleCall(
    trip_id=row["trip_id"],
    stop_sequence=int(row["stop_sequence"]),
    stop_id=row["stop_id"],
    arrival_time=_time(row["arrival_time"]),
    departure_time=_time(row["departure_time"]),
  )


def _calls_by_trip(calls: tuple[ScheduleCall, ...]) -> dict[str, tuple[ScheduleCall, ...]]:
  grouped: dict[str, list[ScheduleCall]] = defaultdict(list)
  for call in calls:
    grouped[call.trip_id].append(call)
  return {trip_id: tuple(sorted(group, key=lambda call: call.stop_sequence)) for trip_id, group in grouped.items()}


def _topology(calls_by_trip: dict[str, tuple[ScheduleCall, ...]]) -> tuple[dict[tuple[str, int], tuple[str, str]], set[tuple[str, str]]]:
  incoming = {}
  edges = set()
  for trip_id, calls in calls_by_trip.items():
    for previous, current in zip(calls, calls[1:]):
      edge = (previous.stop_id, current.stop_id)
      incoming[(trip_id, current.stop_sequence)] = edge
      edges.add(edge)
  return incoming, edges


def _resolve(
  replay: tuple[ReplayRow, ...],
  trips: dict[str, dict[str, str]],
  calls_by_trip: dict[str, tuple[ScheduleCall, ...]],
  stops: set[str],
) -> int:
  calls = {(trip_id, call.stop_sequence): call for trip_id, trip_calls in calls_by_trip.items() for call in trip_calls}
  for row in replay:
    trip = trips.get(row.instance.trip_id)
    call = calls.get((row.instance.trip_id, row.stop_sequence))
    if trip is None or call is None or row.stop_id not in stops:
      raise ReplayError(f"schedule: unresolved identity {row.instance!r} stop_sequence={row.stop_sequence}")
    if (trip["route_id"], int(trip["direction_id"])) != (row.route_id, row.direction_id) or call.stop_id != row.stop_id:
      raise ReplayError(f"schedule: mismatched identity {row.instance!r} stop_sequence={row.stop_sequence}")
    if (call.arrival_time, call.departure_time) != (row.scheduled_arrival_time, row.scheduled_departure_time):
      raise ReplayError(f"schedule: mismatched time {row.instance!r} stop_sequence={row.stop_sequence}")
  return len(replay)


def _headways(replay: tuple[ReplayRow, ...], calls_by_trip: dict[str, tuple[ScheduleCall, ...]]) -> HeadwayAudit:
  next_sequence = {
    (trip_id, current.stop_sequence): following.stop_sequence
    for trip_id, calls in calls_by_trip.items() for current, following in zip(calls, calls[1:])
  }
  trips: dict[tuple[TripInstance, str], list[ReplayRow]] = defaultdict(list)
  for row in replay:
    trips[(row.instance, row.vehicle_id)].append(row)

  departures = []
  for rows in trips.values():
    ordered = sorted(rows, key=lambda row: row.stop_sequence)
    for current, following in zip(ordered, ordered[1:]):
      expected = next_sequence.get((current.instance.trip_id, current.stop_sequence))
      if expected == following.stop_sequence and following.move_timestamp is not None:
        departures.append((current, following.move_timestamp))

  physical: dict[tuple[int, str, str, int, int], list[ReplayRow]] = defaultdict(list)
  for row, target_time in departures:
    physical[(row.instance.service_date, row.vehicle_id, row.parent_station, row.direction_id, target_time)].append(row)
  conflicting = sum(len({row.headway_trunk_seconds for row in rows}) > 1 for rows in physical.values())
  if conflicting:
    raise ReplayError(f"headway: {conflicting} physical departures have conflicting source labels")
  canonical = [(min(rows, key=lambda row: (row.instance, row.stop_sequence)), key[-1]) for key, rows in physical.items()]

  by_lane: dict[tuple[str, str, int], list[tuple[ReplayRow, int]]] = defaultdict(list)
  for row, target_time in canonical:
    by_lane[(row.parent_station, row.trunk_route_id, row.direction_id)].append((row, target_time))
  derived: dict[ReplayRow, tuple[int, int]] = {}
  for events in by_lane.values():
    ordered = sorted(events, key=lambda item: item[1])
    for (_, previous), (row, target_time) in zip(ordered, ordered[1:]):
      value = target_time - previous
      if value <= 0:
        raise ReplayError("headway: departure time did not advance")
      derived[row] = (target_time, value)

  exact = {row for row, (_, value) in derived.items() if row.headway_trunk_seconds == value}
  mismatches = [row for row, (_, value) in derived.items() if row.headway_trunk_seconds is not None and row.headway_trunk_seconds != value]
  if mismatches:
    row = mismatches[0]
    raise ReplayError(f"headway: source mismatch for {row.instance!r} stop_sequence={row.stop_sequence}")
  source = {row for row in replay if row.headway_trunk_seconds is not None}
  derived_only = {row for row in derived if row.headway_trunk_seconds is None}
  bins: dict[tuple[str, int, int], int] = defaultdict(int)
  for row in exact:
    target_time = derived[row][0]
    bins[(row.parent_station, row.direction_id, target_time // BIN_SECONDS * BIN_SECONDS)] += 1

  return HeadwayAudit(
    source_values=len(source),
    derived_values=len(derived),
    exact_matches=len(exact),
    mismatches=0,
    boundary_only=len(source - exact),
    derived_only=len(derived_only),
    physical_departures=len(physical),
    duplicate_aliases=len(departures) - len(physical),
    conflicting_aliases=0,
    parent_stations=len({row.parent_station for row in exact}),
    station_directions=len({(row.parent_station, row.direction_id) for row in exact}),
    target_bins=len(bins),
    colliding_target_bins=sum(count > 1 for count in bins.values()),
    max_targets_per_bin=max(bins.values(), default=0),
  )


def _time(value: str) -> int:
  hour, minute, second = map(int, value.split(":"))
  return hour * 3600 + minute * 60 + second


def _optional_int(value: str) -> int | None:
  return int(value) if value else None


if __name__ == "__main__":
  main()
