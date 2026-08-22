"""Plan and atomically acquire a bounded MBTA LAMP population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from experiments.tools.mbta_replay_extract import SOURCES as REPLAY_SOURCES
from experiments.tools.mbta_replay_extract import USER_AGENT


INDEX_URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/index.csv"
PERFORMANCE_BASE = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1"
START_DATE = date(2026, 7, 24)
END_DATE = date(2026, 8, 20)
CAP_BYTES = 128 * 1024 * 1024
INDEX_CAP_BYTES = 4 * 1024 * 1024
TIMEOUT_SECONDS = 30
INDEX_FIELDS = ("size_bytes", "last_modified", "service_date", "file_url")


class PopulationError(ValueError):
  """The population plan or acquired bytes violate their declared boundary."""


@dataclass(frozen=True)
class Source:
  name: str
  filename: str
  url: str
  bytes: int
  last_modified: str | None
  service_date: str | None
  sha256: str | None


SCHEDULE_EXTRAS = (
  Source(
    "calendar",
    "mbta-2026-calendar.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/calendar.parquet",
    926_587,
    "2026-08-20T02:43:10+00:00",
    None,
    None,
  ),
  Source(
    "calendar_dates",
    "mbta-2026-calendar_dates.parquet",
    "https://performancedata.mbta.com/lamp/gtfs_archive/2026/calendar_dates.parquet",
    145_682,
    "2026-08-20T02:43:11+00:00",
    None,
    None,
  ),
)


@dataclass(frozen=True)
class Plan:
  schema: int
  observed_at: str
  start_date: str
  end_date: str
  cap_bytes: int
  index_url: str
  index_bytes: int
  index_sha256: str
  availability: str
  sources: tuple[Source, ...]

  @property
  def source_bytes(self) -> int:
    return sum(source.bytes for source in self.sources)


def plan(index: bytes, observed_at: datetime) -> Plan:
  """Select one exact complete-date population before source acquisition."""
  if observed_at.tzinfo is None:
    raise PopulationError("index: observed_at must include a timezone")
  try:
    text = index.decode("utf-8")
  except UnicodeDecodeError as error:
    raise PopulationError("index: expected UTF-8 CSV") from error
  reader = csv.DictReader(StringIO(text))
  if tuple(reader.fieldnames or ()) != INDEX_FIELDS:
    raise PopulationError("index: unexpected fields")

  selected: dict[date, Source] = {}
  for line, row in enumerate(reader, start=2):
    try:
      service_date = date.fromisoformat(row["service_date"])
    except (KeyError, ValueError) as error:
      raise PopulationError(f"index:{line}: invalid service_date") from error
    if not START_DATE <= service_date <= END_DATE:
      continue
    if service_date in selected:
      raise PopulationError(f"index:{line}: duplicate service_date {service_date}")
    selected[service_date] = _performance_source(row, line, observed_at)

  expected = tuple(START_DATE + timedelta(days=offset) for offset in range((END_DATE - START_DATE).days + 1))
  missing = tuple(day.isoformat() for day in expected if day not in selected)
  if missing:
    raise PopulationError(f"index: missing service dates {missing!r}")

  schedule = tuple(
    Source(source.name, source.filename, source.url, source.bytes, None, None, source.sha256)
    for source in REPLAY_SOURCES
    if source.name != "performance"
  ) + SCHEDULE_EXTRAS
  sources = tuple(selected[day] for day in expected) + schedule
  total = sum(source.bytes for source in sources)
  if total > CAP_BYTES:
    raise PopulationError(f"plan: declared {total} bytes exceeds cap {CAP_BYTES}")
  return Plan(
    schema=1,
    observed_at=observed_at.isoformat(),
    start_date=START_DATE.isoformat(),
    end_date=END_DATE.isoformat(),
    cap_bytes=CAP_BYTES,
    index_url=INDEX_URL,
    index_bytes=len(index),
    index_sha256=hashlib.sha256(index).hexdigest(),
    availability="retrospective_event_time_only:no_generation_or_ingestion_clock",
    sources=sources,
  )


def fetch_plan(observed_at: datetime) -> Plan:
  """Fetch the bounded official index and return its pure selection plan."""
  request = Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
  with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
    index = response.read(INDEX_CAP_BYTES + 1)
  if len(index) > INDEX_CAP_BYTES:
    raise PopulationError(f"index: exceeds {INDEX_CAP_BYTES} bytes")
  return plan(index, observed_at)


def acquire(value: Plan, directory: Path) -> Plan:
  """Download every planned source into one atomically published directory."""
  _validate_plan(value)
  if directory.exists():
    raise PopulationError(f"acquisition: target already exists {directory}")
  directory.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
  acquired = []
  try:
    for source in value.sources:
      acquired.append(_download(source, staging / source.filename))
    sealed = replace(value, sources=tuple(acquired))
    _write(staging / "manifest.json", sealed)
    staging.replace(directory)
    return sealed
  except BaseException:
    shutil.rmtree(staging)
    raise


def seal(value: Plan, directory: Path) -> Plan:
  """Seal already acquired exact-size files against a refreshed index plan."""
  _validate_plan(value)
  acquired = []
  for source in value.sources:
    path = directory / source.filename
    if not path.is_file() or path.stat().st_size != source.bytes:
      raise PopulationError(f"{source.filename}: artifact drift")
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
      while chunk := artifact.read(1024 * 1024):
        digest.update(chunk)
    actual = digest.hexdigest()
    if source.sha256 is not None and source.sha256 != actual:
      raise PopulationError(f"{source.filename}: checksum drift")
    acquired.append(replace(source, sha256=actual))
  sealed = replace(value, sources=tuple(acquired))
  _write(directory / "manifest.json", sealed)
  return sealed


def audit(directory: Path) -> dict[str, object]:
  """Audit population and exact Schedule identity from sealed local sources."""
  manifest_path = directory / "manifest.json"
  connection, value = open_population(directory)
  feed = str(_source_path(directory, value, "feed_info"))
  event_population, event_groups = audit_events(connection)
  groups = _query(
    connection,
    """
    SELECT service_date, route_id,
      count(*) AS source_rows,
      count(DISTINCT (start_time, trip_id)) AS trip_instances,
      count(DISTINCT vehicle_id) AS vehicles,
      count(*) FILTER (WHERE move_timestamp IS NULL) AS missing_movement,
      count(headway_trunk_seconds) AS source_headways,
      count(*) FILTER (WHERE schedule_resolved) AS schedule_resolved,
      count(*) FILTER (WHERE NOT schedule_resolved) AS schedule_unresolved,
      count(*) FILTER (WHERE NOT schedule_resolved AND source_kind = 'added') AS unresolved_added,
      count(*) FILTER (WHERE NOT schedule_resolved AND source_kind = 'nonrevenue') AS unresolved_nonrevenue,
      count(*) FILTER (WHERE NOT schedule_resolved AND source_kind = 'other') AS unresolved_other
    FROM population
    GROUP BY service_date, route_id
    ORDER BY service_date, route_id
    """,
  )
  event_by_group = {(row["service_date"], row["route_id"]): row for row in event_groups}
  groups = [row | event_by_group[(row["service_date"], row["route_id"])] for row in groups]
  for row in groups:
    row["boundary_only_headways"] = row["source_headways"] - row["exact_headways"] - row["mismatched_headways"]
  versions = _query(
    connection,
    """
    WITH dates AS (SELECT DISTINCT service_date FROM performance)
    SELECT dates.service_date, string_agg(feed.feed_version, ' | ' ORDER BY feed.feed_version) AS feed_version
    FROM dates
    JOIN read_parquet(?) feed
      ON feed.gtfs_active_date <= dates.service_date AND feed.gtfs_end_date >= dates.service_date
    GROUP BY dates.service_date
    ORDER BY dates.service_date
    """,
    [feed],
  )
  totals = _sum_groups(groups)
  sufficient = event_population["exact_headways"] > 0 and event_population["conflicting_aliases"] == 0
  return {
    "schema": 2,
    "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "source_files": len(value.sources),
    "source_bytes": value.source_bytes,
    "start_date": value.start_date,
    "end_date": value.end_date,
    "availability": value.availability,
    "population": {
      "dates": len({row["service_date"] for row in groups}),
      "routes": sorted({row["route_id"] for row in groups}),
      **totals,
    },
    "event_population": event_population,
    "schedule_versions": versions,
    "date_routes": groups,
    "decision": (
      "advance:stage_3_retrospective_with_schedule_mask"
      if sufficient and totals["schedule_unresolved"]
      else "advance:stage_3_retrospective"
      if sufficient
      else "stop:insufficient_event_population"
    ),
  }


def open_population(directory: Path) -> tuple[Any, Plan]:
  """Open one verified source population and its canonical DuckDB tables."""
  value = read(directory / "manifest.json")
  _validate_plan(value)
  for source in value.sources:
    _verify(directory / source.filename, source)
  try:
    import duckdb
  except ImportError as error:
    raise RuntimeError("population audit requires duckdb==1.4.1") from error

  connection = duckdb.connect()
  performance = [str(directory / source.filename) for source in value.sources if source.name == "performance"]
  trips = str(_source_path(directory, value, "trips"))
  calls = str(_source_path(directory, value, "stop_times"))
  stops = str(_source_path(directory, value, "stops"))
  connection.execute("CREATE TEMP TABLE performance AS SELECT * FROM read_parquet(?)", [performance])
  connection.execute(
    """
    CREATE TEMP TABLE schedule_calls AS
    WITH dates AS (SELECT DISTINCT service_date FROM performance),
    calls AS (
      SELECT *,
        split_part(arrival_time, ':', 1)::INT * 3600
          + split_part(arrival_time, ':', 2)::INT * 60
          + split_part(arrival_time, ':', 3)::INT AS arrival,
        split_part(departure_time, ':', 1)::INT * 3600
          + split_part(departure_time, ':', 2)::INT * 60
          + split_part(departure_time, ':', 3)::INT AS departure
      FROM read_parquet(?)
    )
    SELECT DISTINCT dates.service_date, trips.trip_id, trips.route_id, trips.direction_id,
      coalesce(stops.parent_station, stops.stop_id) AS parent_station,
      calls.arrival, calls.departure
    FROM dates
    JOIN read_parquet(?) trips
      ON trips.gtfs_active_date <= dates.service_date AND trips.gtfs_end_date >= dates.service_date
    JOIN calls
      ON calls.trip_id = trips.trip_id
      AND calls.gtfs_active_date <= dates.service_date AND calls.gtfs_end_date >= dates.service_date
    JOIN read_parquet(?) stops
      ON stops.stop_id = calls.stop_id
      AND stops.gtfs_active_date <= dates.service_date AND stops.gtfs_end_date >= dates.service_date
    """,
    [calls, trips, stops],
  )
  connection.execute(
    """
    CREATE TEMP TABLE population AS
    SELECT performance.*,
      schedule_calls.trip_id IS NOT NULL AS schedule_resolved,
      CASE
        WHEN performance.trip_id LIKE 'ADDED-%' THEN 'added'
        WHEN performance.trip_id LIKE 'NONREV-%' THEN 'nonrevenue'
        ELSE 'other'
      END AS source_kind
    FROM performance
    LEFT JOIN schedule_calls
      ON schedule_calls.service_date = performance.service_date
      AND schedule_calls.trip_id = performance.trip_id
      AND schedule_calls.route_id = performance.route_id
      AND schedule_calls.direction_id = performance.direction_id
      AND schedule_calls.parent_station = performance.parent_station
      AND schedule_calls.arrival = performance.scheduled_arrival_time
      AND schedule_calls.departure = performance.scheduled_departure_time
    """
  )
  return connection, value


def audit_events(connection: Any) -> tuple[dict[str, object], list[dict[str, object]]]:
  """Reconstruct the observable event carrier without requiring Schedule identity."""
  connection.execute(
    """
    CREATE TEMP TABLE ambiguous_trips AS
    SELECT DISTINCT service_date, route_id, trip_id
    FROM population
    QUALIFY count(*) OVER (
      PARTITION BY service_date, route_id, trip_id, coalesce(move_timestamp, stop_timestamp)
    ) > 1
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE ordered_population AS
    SELECT population.*,
      coalesce(move_timestamp, stop_timestamp) AS event_order_timestamp,
      row_number() OVER trip AS trip_position,
      lead(stop_id) OVER trip AS following_stop_id,
      lead(stop_sequence) OVER trip AS following_stop_sequence,
      lead(move_timestamp) OVER trip AS departure_timestamp
    FROM population
    ANTI JOIN ambiguous_trips USING (service_date, route_id, trip_id)
    WINDOW trip AS (
      PARTITION BY service_date, route_id, trip_id
      ORDER BY coalesce(move_timestamp, stop_timestamp)
    )
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE event_aliases AS
    SELECT *
    FROM ordered_population
    WHERE stop_count > 1 AND departure_timestamp IS NOT NULL
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE events AS
    SELECT service_date, vehicle_id, parent_station, direction_id, departure_timestamp,
      any_value(trunk_route_id) AS trunk_route_id,
      max(headway_trunk_seconds) AS source_headway_seconds,
      count(*) AS aliases,
      count(DISTINCT trunk_route_id) AS trunk_routes,
      count(DISTINCT headway_trunk_seconds) FILTER (WHERE headway_trunk_seconds IS NOT NULL) AS source_labels
    FROM event_aliases
    GROUP BY service_date, vehicle_id, parent_station, direction_id, departure_timestamp
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE lane_times AS
    SELECT service_date, parent_station, trunk_route_id, direction_id, departure_timestamp,
      count(*) AS event_count
    FROM events
    GROUP BY service_date, parent_station, trunk_route_id, direction_id, departure_timestamp
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE headway_relations AS
    WITH ordered AS (
      SELECT *,
        lag(departure_timestamp) OVER lane AS source_timestamp,
        lag(event_count) OVER lane AS source_count
      FROM lane_times
      WINDOW lane AS (
        PARTITION BY service_date, parent_station, trunk_route_id, direction_id
        ORDER BY departure_timestamp
      )
    )
    SELECT *, departure_timestamp - source_timestamp AS elapsed_seconds
    FROM ordered
    WHERE event_count = 1 AND source_count = 1
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE run_relations AS
    WITH successors AS (
      SELECT *,
        lead(trip_position) OVER trip AS target_position,
        lead(vehicle_id) OVER trip AS target_vehicle_id,
        lead(parent_station) OVER trip AS target_parent_station,
        lead(direction_id) OVER trip AS target_direction_id,
        lead(departure_timestamp) OVER trip AS target_timestamp
      FROM event_aliases
      WINDOW trip AS (
        PARTITION BY service_date, route_id, trip_id
        ORDER BY trip_position
      )
    )
    SELECT DISTINCT service_date, route_id,
      vehicle_id, parent_station, direction_id, departure_timestamp,
      target_vehicle_id, target_parent_station, target_direction_id, target_timestamp
    FROM successors
    WHERE target_position = trip_position + 1 AND departure_timestamp < target_timestamp
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE run_conflict_sources AS
    SELECT service_date, vehicle_id, parent_station, direction_id, departure_timestamp
    FROM run_relations
    GROUP BY ALL
    HAVING count(DISTINCT (target_vehicle_id, target_parent_station, target_direction_id, target_timestamp)) > 1
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE run_conflict_targets AS
    SELECT service_date, target_vehicle_id, target_parent_station, target_direction_id, target_timestamp
    FROM run_relations
    GROUP BY ALL
    HAVING count(DISTINCT (vehicle_id, parent_station, direction_id, departure_timestamp)) > 1
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE exact_run_relations AS
    SELECT run_relations.*
    FROM run_relations
    ANTI JOIN run_conflict_sources USING (
      service_date, vehicle_id, parent_station, direction_id, departure_timestamp
    )
    ANTI JOIN run_conflict_targets USING (
      service_date, target_vehicle_id, target_parent_station, target_direction_id, target_timestamp
    )
    """
  )
  population = _query(
    connection,
    """
    SELECT
      (SELECT count(*) FROM ambiguous_trips) AS ambiguous_order_trips,
      (SELECT count(*) FROM population SEMI JOIN ambiguous_trips USING (service_date, route_id, trip_id)) AS ambiguous_order_rows,
      (SELECT count(*) FROM event_aliases) AS represented_source_rows,
      (SELECT count(*) FROM events) AS physical_departures,
      (SELECT sum(aliases - 1) FROM events) AS duplicate_aliases,
      (SELECT count(*) FROM events WHERE trunk_routes > 1 OR source_labels > 1) AS conflicting_aliases,
      (SELECT count(*) FROM lane_times WHERE event_count > 1) AS simultaneous_groups,
      (SELECT coalesce(sum(event_count), 0) FROM lane_times WHERE event_count > 1) AS simultaneous_events,
      (SELECT count(*) FROM exact_run_relations) AS run_relations,
      (SELECT count(*) FROM run_conflict_sources) AS ambiguous_run_sources,
      (SELECT count(*) FROM run_conflict_targets) AS ambiguous_run_targets,
      (SELECT count(*) FROM headway_relations) AS headway_relations,
      (SELECT count(*) FROM event_aliases aliases JOIN headway_relations relations
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
        WHERE aliases.headway_trunk_seconds = relations.elapsed_seconds) AS exact_headways,
      (SELECT count(*) FROM event_aliases aliases JOIN headway_relations relations
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
        WHERE aliases.headway_trunk_seconds IS NOT NULL
          AND aliases.headway_trunk_seconds != relations.elapsed_seconds) AS mismatched_headways,
      (SELECT count(*) FROM headway_relations relations JOIN events
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
        WHERE events.source_headway_seconds IS NULL) AS derived_only_headways,
      (SELECT count(headway_trunk_seconds) FROM population)
        - (SELECT count(*) FROM event_aliases aliases JOIN headway_relations relations
          USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
          WHERE aliases.headway_trunk_seconds = relations.elapsed_seconds)
        - (SELECT count(*) FROM event_aliases aliases JOIN headway_relations relations
          USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
          WHERE aliases.headway_trunk_seconds IS NOT NULL
            AND aliases.headway_trunk_seconds != relations.elapsed_seconds) AS boundary_only_headways,
      (SELECT count(DISTINCT (parent_station, trunk_route_id, direction_id)) FROM events) AS station_directions,
      (SELECT min(elapsed_seconds) FROM headway_relations) AS minimum_gap_seconds,
      (SELECT quantile_disc(elapsed_seconds, 0.5) FROM headway_relations) AS median_gap_seconds,
      (SELECT quantile_disc(elapsed_seconds, 0.95) FROM headway_relations) AS p95_gap_seconds,
      (SELECT max(elapsed_seconds) FROM headway_relations) AS maximum_gap_seconds
    """,
  )[0]
  groups = _query(
    connection,
    """
    WITH date_routes AS (
      SELECT DISTINCT service_date, route_id FROM population
    ), carrier AS (
      SELECT service_date, route_id,
        count(*) AS represented_source_rows,
        count(DISTINCT (vehicle_id, parent_station, direction_id, departure_timestamp)) AS physical_departures,
        count(*) - count(DISTINCT (vehicle_id, parent_station, direction_id, departure_timestamp)) AS duplicate_aliases,
        count(DISTINCT (parent_station, trunk_route_id, direction_id)) AS lanes,
        min(departure_timestamp) AS first_departure_timestamp,
        max(departure_timestamp) AS last_departure_timestamp
      FROM event_aliases
      GROUP BY service_date, route_id
    ), labels AS (
      SELECT aliases.service_date, aliases.route_id,
        count(*) FILTER (WHERE aliases.headway_trunk_seconds = relations.elapsed_seconds) AS exact_headways,
        count(*) FILTER (
          WHERE aliases.headway_trunk_seconds IS NOT NULL
            AND aliases.headway_trunk_seconds != relations.elapsed_seconds
        ) AS mismatched_headways,
        count(*) FILTER (WHERE aliases.headway_trunk_seconds IS NULL) AS derived_only_headways,
        count(*) AS headway_relations,
        quantile_disc(relations.elapsed_seconds, 0.5) AS median_gap_seconds,
        quantile_disc(relations.elapsed_seconds, 0.95) AS p95_gap_seconds,
        max(relations.elapsed_seconds) AS maximum_gap_seconds
      FROM event_aliases aliases
      JOIN headway_relations relations
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
      GROUP BY aliases.service_date, aliases.route_id
    ), ambiguous AS (
      SELECT population.service_date, population.route_id,
        count(DISTINCT population.trip_id) AS ambiguous_order_trips,
        count(*) AS ambiguous_order_rows
      FROM population
      JOIN ambiguous_trips USING (service_date, route_id, trip_id)
      GROUP BY population.service_date, population.route_id
    ), runs AS (
      SELECT service_date, route_id, count(*) AS run_relations
      FROM exact_run_relations
      GROUP BY service_date, route_id
    )
    SELECT date_routes.service_date, date_routes.route_id,
      coalesce(carrier.represented_source_rows, 0) AS represented_source_rows,
      coalesce(carrier.physical_departures, 0) AS physical_departures,
      coalesce(carrier.duplicate_aliases, 0) AS duplicate_aliases,
      coalesce(carrier.lanes, 0) AS lanes,
      coalesce(carrier.first_departure_timestamp, 0) AS first_departure_timestamp,
      coalesce(carrier.last_departure_timestamp, 0) AS last_departure_timestamp,
      coalesce(ambiguous.ambiguous_order_trips, 0) AS ambiguous_order_trips,
      coalesce(ambiguous.ambiguous_order_rows, 0) AS ambiguous_order_rows,
      coalesce(runs.run_relations, 0) AS run_relations,
      coalesce(labels.headway_relations, 0) AS headway_relations,
      coalesce(labels.exact_headways, 0) AS exact_headways,
      coalesce(labels.mismatched_headways, 0) AS mismatched_headways,
      coalesce(labels.derived_only_headways, 0) AS derived_only_headways,
      coalesce(labels.median_gap_seconds, 0) AS median_gap_seconds,
      coalesce(labels.p95_gap_seconds, 0) AS p95_gap_seconds,
      coalesce(labels.maximum_gap_seconds, 0) AS maximum_gap_seconds
    FROM date_routes
    LEFT JOIN carrier USING (service_date, route_id)
    LEFT JOIN labels USING (service_date, route_id)
    LEFT JOIN ambiguous USING (service_date, route_id)
    LEFT JOIN runs USING (service_date, route_id)
    ORDER BY date_routes.service_date, date_routes.route_id
    """,
  )
  return population, groups


def read(path: Path) -> Plan:
  value = json.loads(path.read_text())
  try:
    return Plan(
      schema=value["schema"],
      observed_at=value["observed_at"],
      start_date=value["start_date"],
      end_date=value["end_date"],
      cap_bytes=value["cap_bytes"],
      index_url=value["index_url"],
      index_bytes=value["index_bytes"],
      index_sha256=value["index_sha256"],
      availability=value["availability"],
      sources=tuple(Source(**source) for source in value["sources"]),
    )
  except (KeyError, TypeError) as error:
    raise PopulationError(f"manifest: invalid {path}") from error


def _performance_source(row: dict[str, str], line: int, observed_at: datetime) -> Source:
  service_date = date.fromisoformat(row["service_date"])
  filename = f"{service_date}-subway-on-time-performance-v1.parquet"
  url = f"{PERFORMANCE_BASE}/{filename}"
  try:
    size = int(row["size_bytes"])
    modified = datetime.fromisoformat(row["last_modified"])
  except (KeyError, ValueError) as error:
    raise PopulationError(f"index:{line}: invalid typed value") from error
  if size <= 0 or row["file_url"] != url:
    raise PopulationError(f"index:{line}: invalid source identity {service_date}")
  if modified.tzinfo is None or modified > observed_at:
    raise PopulationError(f"index:{line}: impossible publication time {service_date}")
  local_day = observed_at.astimezone(ZoneInfo("America/New_York")).date()
  if service_date >= local_day:
    raise PopulationError(f"index:{line}: incomplete service date {service_date}")
  return Source("performance", filename, url, size, modified.isoformat(), service_date.isoformat(), None)


def _validate_plan(value: Plan) -> None:
  if value.schema != 1 or value.cap_bytes != CAP_BYTES:
    raise PopulationError("plan: unsupported contract")
  if value.source_bytes > value.cap_bytes:
    raise PopulationError(f"plan: declared {value.source_bytes} bytes exceeds cap {value.cap_bytes}")
  filenames = tuple(source.filename for source in value.sources)
  if len(filenames) != len(set(filenames)):
    raise PopulationError("plan: duplicate filename")
  performance = tuple(source for source in value.sources if source.name == "performance")
  if len(performance) != 28 or (performance[0].service_date, performance[-1].service_date) != (
    value.start_date,
    value.end_date,
  ):
    raise PopulationError("plan: population drift")


def _download(source: Source, path: Path) -> Source:
  request = Request(source.url, headers={"User-Agent": USER_AGENT})
  digest = hashlib.sha256()
  remaining = source.bytes
  with urlopen(request, timeout=TIMEOUT_SECONDS) as response, path.open("xb") as output:
    while chunk := response.read(min(1024 * 1024, remaining + 1)):
      remaining -= len(chunk)
      if remaining < 0:
        raise PopulationError(f"{source.filename}: exceeds declared {source.bytes} bytes")
      digest.update(chunk)
      output.write(chunk)
  if remaining:
    raise PopulationError(f"{source.filename}: expected {source.bytes} bytes, got {source.bytes - remaining}")
  actual = digest.hexdigest()
  if source.sha256 is not None and actual != source.sha256:
    raise PopulationError(f"{source.filename}: checksum drift")
  return replace(source, sha256=actual)


def _verify(path: Path, source: Source) -> None:
  if source.sha256 is None:
    raise PopulationError(f"{source.filename}: unsealed checksum")
  if not path.is_file() or path.stat().st_size != source.bytes:
    raise PopulationError(f"{source.filename}: artifact drift")
  digest = hashlib.sha256()
  with path.open("rb") as artifact:
    while chunk := artifact.read(1024 * 1024):
      digest.update(chunk)
  if digest.hexdigest() != source.sha256:
    raise PopulationError(f"{source.filename}: checksum drift")


def _source_path(directory: Path, value: Plan, name: str) -> Path:
  source = next((source for source in value.sources if source.name == name), None)
  if source is None:
    raise PopulationError(f"manifest: missing {name}")
  return directory / source.filename


def _query(connection: Any, query: str, parameters: list[str] | None = None) -> list[dict[str, object]]:
  cursor = connection.execute(query, parameters or [])
  columns = tuple(column[0] for column in cursor.description)
  return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _sum_groups(groups: list[dict[str, object]]) -> dict[str, int]:
  fields = (
    "source_rows",
    "trip_instances",
    "missing_movement",
    "source_headways",
    "schedule_resolved",
    "schedule_unresolved",
    "unresolved_added",
    "unresolved_nonrevenue",
    "unresolved_other",
  )
  return {field: sum(int(row[field]) for row in groups) for field in fields}


def _write(path: Path, value: Plan) -> None:
  _write_json(path, asdict(value))


def _write_json(path: Path, value: object) -> None:
  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser()
  commands = parser.add_subparsers(dest="command", required=True)
  planner = commands.add_parser("plan")
  planner.add_argument("--output", type=Path, required=True)
  planner.add_argument("--observed-at", type=datetime.fromisoformat, required=True)
  acquisition = commands.add_parser("acquire")
  acquisition.add_argument("--plan", type=Path, required=True)
  acquisition.add_argument("--source-dir", type=Path, required=True)
  sealer = commands.add_parser("seal")
  sealer.add_argument("--plan", type=Path, required=True)
  sealer.add_argument("--source-dir", type=Path, required=True)
  auditor = commands.add_parser("audit")
  auditor.add_argument("--source-dir", type=Path, required=True)
  auditor.add_argument("--output", type=Path, required=True)
  recorder = commands.add_parser("record")
  recorder.add_argument("--source-dir", type=Path, required=True)
  recorder.add_argument("--output-dir", type=Path, required=True)
  arguments = parser.parse_args()
  if arguments.command == "plan":
    value = fetch_plan(arguments.observed_at)
    _write(arguments.output, value)
    print(json.dumps({"sources": len(value.sources), "bytes": value.source_bytes, "cap_bytes": value.cap_bytes}))
  elif arguments.command == "acquire":
    value = acquire(read(arguments.plan), arguments.source_dir)
    print(json.dumps({"sources": len(value.sources), "bytes": value.source_bytes, "manifest": str(arguments.source_dir / "manifest.json")}))
  elif arguments.command == "seal":
    value = seal(read(arguments.plan), arguments.source_dir)
    print(json.dumps({"sources": len(value.sources), "bytes": value.source_bytes, "manifest": str(arguments.source_dir / "manifest.json")}))
  elif arguments.command == "audit":
    result = audit(arguments.source_dir)
    _write_json(arguments.output, result)
    print(json.dumps({"decision": result["decision"], "output": str(arguments.output)}))
  else:
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read(arguments.source_dir / "manifest.json")
    result = audit(arguments.source_dir)
    _write(arguments.output_dir / "manifest.json", manifest)
    _write_json(arguments.output_dir / "audit.json", result)
    print(json.dumps({"decision": result["decision"], "output": str(arguments.output_dir)}))


if __name__ == "__main__":
  main()
