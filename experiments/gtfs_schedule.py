"""Validate the pinned fixed-stop GTFS Schedule witness."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


REVISION = "474750a163088673df718838d4a1bb093391f9af"
SOURCE_ID = "google-transit/sample-feed-1"
SOURCE_URL = (
  "https://raw.githubusercontent.com/google/transit/"
  f"{REVISION}/gtfs/spec/en/examples/sample-feed-1.zip"
)
SOURCE_SHA256 = "46404f91b8f852bf1037f7901abc215d91d1bbe2a4362c5b02a3f9ced53c8d65"
SOURCE_MAX_BYTES = 16 * 1024
SOURCE_TIMEOUT_SECONDS = 10
ARCHIVE_MAX_BYTES = 32 * 1024
REQUIRED_FILES = {
  "agency.txt",
  "calendar.txt",
  "calendar_dates.txt",
  "frequencies.txt",
  "routes.txt",
  "stop_times.txt",
  "stops.txt",
  "trips.txt",
}
KNOWN_UNUSED_FILES = {"fare_attributes.txt", "fare_rules.txt", "shapes.txt"}
TIME = re.compile(r"^(\d+):(\d{2}):(\d{2})$")


class ScheduleError(ValueError):
  """A contextual GTFS Schedule boundary failure."""


@dataclass(frozen=True)
class SourceRow:
  file: str
  line: int


@dataclass(frozen=True)
class Agency:
  agency_id: str
  timezone: str
  source: SourceRow


@dataclass(frozen=True)
class Service:
  service_id: str
  weekdays: tuple[bool, ...]
  start_date: date
  end_date: date
  source: SourceRow


@dataclass(frozen=True)
class ServiceException:
  service_id: str
  service_date: date
  added: bool
  source: SourceRow


@dataclass(frozen=True)
class Stop:
  stop_id: str
  name: str
  latitude: float
  longitude: float
  source: SourceRow


@dataclass(frozen=True)
class Route:
  route_id: str
  agency_id: str
  short_name: str
  long_name: str
  route_type: int
  source: SourceRow


@dataclass(frozen=True)
class Trip:
  trip_id: str
  route_id: str
  service_id: str
  source: SourceRow


@dataclass(frozen=True)
class StopTime:
  trip_id: str
  arrival: int | None
  departure: int | None
  stop_id: str
  stop_sequence: int
  source: SourceRow


@dataclass(frozen=True)
class Frequency:
  trip_id: str
  start: int
  end: int
  headway_seconds: int
  exact_times: int
  source: SourceRow


@dataclass(frozen=True)
class SegmentOccurrence:
  trip_id: str
  from_sequence: int
  to_sequence: int


@dataclass(frozen=True)
class RouteSegment:
  route_id: str
  from_stop_id: str
  to_stop_id: str
  occurrences: tuple[SegmentOccurrence, ...]


@dataclass(frozen=True)
class Schedule:
  source_id: str
  revision: str
  sha256: str
  agencies: tuple[Agency, ...]
  services: tuple[Service, ...]
  exceptions: tuple[ServiceException, ...]
  stops: tuple[Stop, ...]
  routes: tuple[Route, ...]
  trips: tuple[Trip, ...]
  stop_times: tuple[StopTime, ...]
  frequencies: tuple[Frequency, ...]
  ordered_calls: tuple[tuple[str, tuple[StopTime, ...]], ...]
  segments: tuple[RouteSegment, ...]


@dataclass(frozen=True)
class Observation:
  source_id: str
  revision: str
  sha256: str
  agencies: int
  services: int
  exceptions: int
  stops: int
  routes: int
  trips: int
  stop_times: int
  frequencies: int
  segments: int
  post_midnight_times: int


def load(path: str | Path | None = None) -> Schedule:
  """Load the exact pinned sample from a local override or its revision URL."""
  data = _read_source(path)
  digest = hashlib.sha256(data).hexdigest()
  if digest != SOURCE_SHA256:
    raise ScheduleError(f"source: sha256 expected {SOURCE_SHA256}, got {digest}")
  return parse(data, source_id=SOURCE_ID, revision=REVISION, sha256=digest)


def parse(data: bytes, *, source_id: str = "fixture", revision: str = "fixture", sha256: str | None = None) -> Schedule:
  """Parse bounded GTFS ZIP bytes for local deterministic fixtures."""
  if len(data) > SOURCE_MAX_BYTES:
    raise ScheduleError(f"source: exceeds {SOURCE_MAX_BYTES} bytes")
  digest = sha256 or hashlib.sha256(data).hexdigest()
  try:
    with ZipFile(io.BytesIO(data)) as archive:
      tables = _archive_tables(archive)
  except BadZipFile as error:
    raise ScheduleError("source: expected a GTFS ZIP archive") from error

  agencies = _agencies(tables["agency.txt"])
  services = _services(tables["calendar.txt"])
  exceptions = _exceptions(tables["calendar_dates.txt"])
  stops = _stops(tables["stops.txt"])
  routes = _routes(tables["routes.txt"])
  trips = _trips(tables["trips.txt"])
  stop_times = _stop_times(tables["stop_times.txt"])
  frequencies = _frequencies(tables["frequencies.txt"])
  _validate(agencies, services, exceptions, stops, routes, trips, stop_times, frequencies)
  ordered_calls = _ordered_calls(trips, stop_times)
  segments = _segments(trips, ordered_calls)
  return Schedule(
    source_id,
    revision,
    digest,
    agencies,
    services,
    exceptions,
    stops,
    routes,
    trips,
    stop_times,
    frequencies,
    ordered_calls,
    segments,
  )


def observe(path: str | Path | None = None) -> Observation:
  schedule = load(path)
  times = (
    value
    for stop_time in schedule.stop_times
    for value in (stop_time.arrival, stop_time.departure)
    if value is not None
  )
  return Observation(
    schedule.source_id,
    schedule.revision,
    schedule.sha256,
    len(schedule.agencies),
    len(schedule.services),
    len(schedule.exceptions),
    len(schedule.stops),
    len(schedule.routes),
    len(schedule.trips),
    len(schedule.stop_times),
    len(schedule.frequencies),
    len(schedule.segments),
    sum(value >= 24 * 3600 for value in times),
  )


def main() -> None:
  if len(sys.argv) > 2:
    raise SystemExit("usage: python -m experiments.gtfs_schedule [sample-feed-1.zip]")
  path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
  print(json.dumps(asdict(observe(path)), indent=2))


def _read_source(path: str | Path | None) -> bytes:
  if path is not None:
    source = Path(path)
    if source.stat().st_size > SOURCE_MAX_BYTES:
      raise ScheduleError(f"source: exceeds {SOURCE_MAX_BYTES} bytes")
    return source.read_bytes()
  try:
    with urlopen(Request(SOURCE_URL, headers={"User-Agent": "tinymesh-gtfs-study"}), timeout=SOURCE_TIMEOUT_SECONDS) as response:
      length = response.headers.get("Content-Length")
      if length is not None and int(length) > SOURCE_MAX_BYTES:
        raise ScheduleError(f"source: exceeds {SOURCE_MAX_BYTES} bytes")
      data = response.read(SOURCE_MAX_BYTES + 1)
  except OSError as error:
    raise ScheduleError(f"source: acquisition failed: {error}") from error
  if len(data) > SOURCE_MAX_BYTES:
    raise ScheduleError(f"source: exceeds {SOURCE_MAX_BYTES} bytes")
  return data


def _archive_tables(archive: ZipFile) -> dict[str, tuple[dict[str, str], ...]]:
  names = archive.namelist()
  if len(names) != len(set(names)):
    raise ScheduleError("source: duplicate ZIP member")
  unknown = set(names) - REQUIRED_FILES - KNOWN_UNUSED_FILES
  missing = REQUIRED_FILES - set(names)
  if unknown:
    raise ScheduleError(f"source: unsupported ZIP member {sorted(unknown)[0]!r}")
  if missing:
    raise ScheduleError(f"source: missing {sorted(missing)[0]}")
  if sum(info.file_size for info in archive.infolist()) > ARCHIVE_MAX_BYTES:
    raise ScheduleError(f"source: uncompressed archive exceeds {ARCHIVE_MAX_BYTES} bytes")
  return {name: _rows(name, archive.read(name)) for name in REQUIRED_FILES}


def _rows(name: str, data: bytes) -> tuple[dict[str, str], ...]:
  try:
    text = data.decode("utf-8-sig")
  except UnicodeDecodeError as error:
    raise ScheduleError(f"{name}: invalid UTF-8") from error
  reader = csv.DictReader(io.StringIO(text, newline=""))
  fields = reader.fieldnames
  if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
    raise ScheduleError(f"{name}: invalid header")
  rows = []
  for line, row in enumerate(reader, start=2):
    if None in row:
      raise ScheduleError(f"{name}:{line}: malformed CSV row")
    rows.append({key: (value or "").strip() for key, value in row.items()})
  return tuple(rows)


def _required(row: dict[str, str], field: str, source: SourceRow) -> str:
  value = row.get(field, "")
  if not value:
    _fail(source, field, "required")
  return value


def _integer(row: dict[str, str], field: str, source: SourceRow, *, minimum: int = 0) -> int:
  value = _required(row, field, source)
  try:
    number = int(value)
  except ValueError:
    _fail(source, field, f"expected integer, got {value!r}")
  if number < minimum:
    _fail(source, field, f"expected at least {minimum}, got {number}")
  return number


def _float(row: dict[str, str], field: str, source: SourceRow, low: float, high: float) -> float:
  value = _required(row, field, source)
  try:
    number = float(value)
  except ValueError:
    _fail(source, field, f"expected number, got {value!r}")
  if not low <= number <= high:
    _fail(source, field, f"expected [{low}, {high}], got {number}")
  return number


def _time(value: str, source: SourceRow, field: str, *, optional: bool = False) -> int | None:
  if not value and optional:
    return None
  match = TIME.fullmatch(value)
  if match is None:
    _fail(source, field, f"invalid service-day time {value!r}")
  hour, minute, second = map(int, match.groups())
  if minute >= 60 or second >= 60:
    _fail(source, field, f"invalid service-day time {value!r}")
  return hour * 3600 + minute * 60 + second


def _date(value: str, source: SourceRow, field: str) -> date:
  if len(value) != 8 or not value.isdigit():
    _fail(source, field, f"invalid date {value!r}")
  try:
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))
  except ValueError:
    _fail(source, field, f"invalid date {value!r}")


def _source(file: str, index: int) -> SourceRow:
  return SourceRow(file, index + 2)


def _agencies(rows: tuple[dict[str, str], ...]) -> tuple[Agency, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("agency.txt", index)
    result.append(Agency(_required(row, "agency_id", source), _required(row, "agency_timezone", source), source))
  return tuple(result)


def _services(rows: tuple[dict[str, str], ...]) -> tuple[Service, ...]:
  result = []
  days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
  for index, row in enumerate(rows):
    source = _source("calendar.txt", index)
    values = tuple(_integer(row, day, source) for day in days)
    if any(value not in (0, 1) for value in values):
      _fail(source, "weekday", "expected 0 or 1")
    start = _date(_required(row, "start_date", source), source, "start_date")
    end = _date(_required(row, "end_date", source), source, "end_date")
    if end < start:
      _fail(source, "end_date", "precedes start_date")
    result.append(Service(_required(row, "service_id", source), tuple(bool(value) for value in values), start, end, source))
  return tuple(result)


def _exceptions(rows: tuple[dict[str, str], ...]) -> tuple[ServiceException, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("calendar_dates.txt", index)
    kind = _integer(row, "exception_type", source, minimum=1)
    if kind not in (1, 2):
      _fail(source, "exception_type", "expected 1 or 2")
    result.append(ServiceException(_required(row, "service_id", source), _date(_required(row, "date", source), source, "date"), kind == 1, source))
  return tuple(result)


def _stops(rows: tuple[dict[str, str], ...]) -> tuple[Stop, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("stops.txt", index)
    location_type = row.get("location_type", "")
    if location_type not in ("", "0"):
      _fail(source, "location_type", "only stops/platforms are supported")
    result.append(
      Stop(
        _required(row, "stop_id", source),
        _required(row, "stop_name", source),
        _float(row, "stop_lat", source, -90, 90),
        _float(row, "stop_lon", source, -180, 180),
        source,
      )
    )
  return tuple(result)


def _routes(rows: tuple[dict[str, str], ...]) -> tuple[Route, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("routes.txt", index)
    result.append(
      Route(
        _required(row, "route_id", source),
        _required(row, "agency_id", source),
        row.get("route_short_name", ""),
        row.get("route_long_name", ""),
        _integer(row, "route_type", source),
        source,
      )
    )
  return tuple(result)


def _trips(rows: tuple[dict[str, str], ...]) -> tuple[Trip, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("trips.txt", index)
    result.append(
      Trip(
        _required(row, "trip_id", source),
        _required(row, "route_id", source),
        _required(row, "service_id", source),
        source,
      )
    )
  return tuple(result)


def _stop_times(rows: tuple[dict[str, str], ...]) -> tuple[StopTime, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("stop_times.txt", index)
    if row.get("location_group_id", "") or row.get("location_id", ""):
      _fail(source, "stop_id", "flexible locations are unsupported")
    arrival = _time(row.get("arrival_time", ""), source, "arrival_time", optional=True)
    departure = _time(row.get("departure_time", ""), source, "departure_time", optional=True)
    if arrival is not None and departure is not None and departure < arrival:
      _fail(source, "departure_time", "precedes arrival_time")
    result.append(
      StopTime(
        _required(row, "trip_id", source),
        arrival,
        departure,
        _required(row, "stop_id", source),
        _integer(row, "stop_sequence", source),
        source,
      )
    )
  return tuple(result)


def _frequencies(rows: tuple[dict[str, str], ...]) -> tuple[Frequency, ...]:
  result = []
  for index, row in enumerate(rows):
    source = _source("frequencies.txt", index)
    start = _time(_required(row, "start_time", source), source, "start_time")
    end = _time(_required(row, "end_time", source), source, "end_time")
    assert start is not None and end is not None
    if end <= start:
      _fail(source, "end_time", "must follow start_time")
    exact = _integer(row, "exact_times", source) if row.get("exact_times", "") else 0
    if exact not in (0, 1):
      _fail(source, "exact_times", "expected 0 or 1")
    result.append(Frequency(_required(row, "trip_id", source), start, end, _integer(row, "headway_secs", source, minimum=1), exact, source))
  return tuple(result)


def _validate(
  agencies: tuple[Agency, ...],
  services: tuple[Service, ...],
  exceptions: tuple[ServiceException, ...],
  stops: tuple[Stop, ...],
  routes: tuple[Route, ...],
  trips: tuple[Trip, ...],
  stop_times: tuple[StopTime, ...],
  frequencies: tuple[Frequency, ...],
) -> None:
  agency_ids = _unique("agency_id", agencies, lambda item: item.agency_id)
  service_ids = _unique("service_id", services, lambda item: item.service_id)
  stop_ids = _unique("stop_id", stops, lambda item: item.stop_id)
  route_ids = _unique("route_id", routes, lambda item: item.route_id)
  trip_ids = _unique("trip_id", trips, lambda item: item.trip_id)
  _unique("service exception", exceptions, lambda item: (item.service_id, item.service_date))
  _unique("stop time", stop_times, lambda item: (item.trip_id, item.stop_sequence))
  for route in routes:
    _reference(route.source, "agency_id", route.agency_id, agency_ids)
  for trip in trips:
    _reference(trip.source, "route_id", trip.route_id, route_ids)
    _reference(trip.source, "service_id", trip.service_id, service_ids)
  for exception in exceptions:
    _reference(exception.source, "service_id", exception.service_id, service_ids)
  calls: dict[str, list[StopTime]] = defaultdict(list)
  for stop_time in stop_times:
    _reference(stop_time.source, "trip_id", stop_time.trip_id, trip_ids)
    _reference(stop_time.source, "stop_id", stop_time.stop_id, stop_ids)
    calls[stop_time.trip_id].append(stop_time)
  for trip in trips:
    ordered = sorted(calls[trip.trip_id], key=lambda item: item.stop_sequence)
    if len(ordered) < 2:
      _fail(trip.source, "trip_id", "requires at least two stop times")
    for stop_time in (ordered[0], ordered[-1]):
      if stop_time.arrival is None or stop_time.departure is None:
        _fail(stop_time.source, "arrival_time", "first and last calls require arrival and departure")
  by_trip: dict[str, list[Frequency]] = defaultdict(list)
  for frequency in frequencies:
    _reference(frequency.source, "trip_id", frequency.trip_id, trip_ids)
    by_trip[frequency.trip_id].append(frequency)
  for rows in by_trip.values():
    ordered = sorted(rows, key=lambda item: (item.start, item.end))
    for left, right in zip(ordered, ordered[1:]):
      if right.start < left.end:
        _fail(right.source, "start_time", "frequency intervals overlap")


def _ordered_calls(trips: tuple[Trip, ...], stop_times: tuple[StopTime, ...]) -> tuple[tuple[str, tuple[StopTime, ...]], ...]:
  calls: dict[str, list[StopTime]] = defaultdict(list)
  for stop_time in stop_times:
    calls[stop_time.trip_id].append(stop_time)
  return tuple(
    (trip.trip_id, tuple(sorted(calls[trip.trip_id], key=lambda item: item.stop_sequence)))
    for trip in sorted(trips, key=lambda item: item.trip_id)
  )


def _segments(
  trips: tuple[Trip, ...], ordered_calls: tuple[tuple[str, tuple[StopTime, ...]], ...]
) -> tuple[RouteSegment, ...]:
  route_by_trip = {trip.trip_id: trip.route_id for trip in trips}
  occurrences: dict[tuple[str, str, str], list[SegmentOccurrence]] = defaultdict(list)
  for trip_id, calls in ordered_calls:
    for left, right in zip(calls, calls[1:]):
      key = (route_by_trip[trip_id], left.stop_id, right.stop_id)
      occurrences[key].append(SegmentOccurrence(trip_id, left.stop_sequence, right.stop_sequence))
  return tuple(RouteSegment(*key, tuple(values)) for key, values in sorted(occurrences.items()))


def _unique(name: str, values: tuple, key) -> set:
  seen = set()
  for item in values:
    value = key(item)
    if value in seen:
      _fail(item.source, name, f"duplicate {value!r}")
    seen.add(value)
  return seen


def _reference(source: SourceRow, field: str, value: str, owners: set) -> None:
  if value not in owners:
    _fail(source, field, f"unknown reference {value!r}")


def _fail(source: SourceRow, field: str, reason: str):
  raise ScheduleError(f"{source.file}:{source.line}:{field}: {reason}")


if __name__ == "__main__":
  main()
