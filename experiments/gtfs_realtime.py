"""Normalize one declared GTFS Realtime full-snapshot witness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from experiments.gtfs_schedule import Schedule, StopTime, Trip, load as load_schedule


FIXTURE = Path(__file__).parent / "fixtures" / "gtfs_realtime.json"
SUPPORTED_RELATIONSHIPS = {"SCHEDULED", "CANCELED"}
SUPPORTED_VEHICLE_STATES = {"INCOMING_AT", "STOPPED_AT", "IN_TRANSIT_TO"}


class RealtimeError(ValueError):
  """A contextual GTFS Realtime boundary failure."""


@dataclass(frozen=True, order=True)
class TripInstance:
  source_id: str
  trip_id: str
  service_date: date
  scheduled_start: int


@dataclass(frozen=True)
class EventPrediction:
  delay_seconds: int | None
  predicted_at: int | None


@dataclass(frozen=True)
class StopPrediction:
  stop_id: str
  stop_sequence: int
  arrival: EventPrediction | None
  departure: EventPrediction | None


@dataclass(frozen=True)
class TripObservation:
  entity_id: str
  instance: TripInstance
  route_id: str
  relationship: str
  observed_at: int
  delay_seconds: int | None
  predictions: tuple[StopPrediction, ...]


@dataclass(frozen=True)
class VehicleObservation:
  entity_id: str
  vehicle_id: str
  instance: TripInstance
  observed_at: int
  current_stop_id: str
  current_stop_sequence: int
  current_status: str


@dataclass(frozen=True)
class Snapshot:
  source_id: str
  schedule_revision: str
  schedule_sha256: str
  generated_at: int
  trips: tuple[TripObservation, ...]
  vehicles: tuple[VehicleObservation, ...]


@dataclass(frozen=True)
class Observation:
  source_id: str
  generated_at: int
  trips: int
  vehicles: int
  cancellations: int
  predictions: int


def normalize(message: object, schedule: Schedule, source_id: str) -> Snapshot:
  """Validate one full message and return all-or-nothing immutable facts."""
  if not source_id.strip():
    raise RealtimeError("source_id: required acquisition provenance")
  if source_id != schedule.source_id:
    raise RealtimeError(f"source_id: expected schedule namespace {schedule.source_id!r}, got {source_id!r}")
  root = _object(message, "message")
  header = _object(root.get("header"), "header")
  if _string(header, "incrementality", "header") != "FULL_DATASET":
    raise RealtimeError("header.incrementality: only FULL_DATASET is supported")
  generated_at = _integer(header, "timestamp", "header", minimum=1)
  entities = _list(root, "entities", "message")
  identified = [(_string(_object(value, f"entities[{index}]"), "id", f"entities[{index}]"), value) for index, value in enumerate(entities)]
  _duplicates("entity id", (identity for identity, _ in identified))

  trips = []
  vehicles = []
  for entity_id, value in sorted(identified):
    trip, vehicle = _entity(_object(value, f"entity {entity_id!r}"), entity_id, schedule, source_id, generated_at)
    trips.append(trip)
    if vehicle is not None:
      vehicles.append(vehicle)
  _contradictions(trips, vehicles)
  return Snapshot(
    source_id,
    schedule.revision,
    schedule.sha256,
    generated_at,
    tuple(trips),
    tuple(sorted(vehicles, key=lambda item: item.vehicle_id)),
  )


def observe(path: str | Path = FIXTURE) -> Observation:
  message = json.loads(Path(path).read_text())
  schedule = load_schedule()
  snapshot = normalize(message, schedule, schedule.source_id)
  return Observation(
    snapshot.source_id,
    snapshot.generated_at,
    len(snapshot.trips),
    len(snapshot.vehicles),
    sum(trip.relationship == "CANCELED" for trip in snapshot.trips),
    sum(len(trip.predictions) for trip in snapshot.trips),
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _entity(
  entity: dict,
  entity_id: str,
  schedule: Schedule,
  source_id: str,
  generated_at: int,
) -> tuple[TripObservation, VehicleObservation | None]:
  context = f"entity {entity_id!r}"
  if entity.get("is_deleted"):
    raise RealtimeError(f"{context}: is_deleted is unsupported in a full message")
  if "alert" in entity:
    raise RealtimeError(f"{context}: alerts are unsupported")
  descriptor = _object(entity.get("trip"), f"{context}.trip")
  relationship = _string(descriptor, "schedule_relationship", f"{context}.trip")
  if relationship not in SUPPORTED_RELATIONSHIPS:
    raise RealtimeError(f"{context}.trip.schedule_relationship: unsupported {relationship!r}")
  instance = _instance(descriptor, source_id, context)
  trip, calls = _resolve(instance, descriptor, schedule, context)
  route_id = _string(descriptor, "route_id", f"{context}.trip")

  if relationship == "CANCELED":
    if "vehicle" in entity or "trip_update" in entity:
      raise RealtimeError(f"{context}: canceled trip contradicts active observations")
    return TripObservation(entity_id, instance, route_id, relationship, generated_at, None, ()), None

  vehicle = _vehicle(entity, entity_id, instance, calls, generated_at, context)
  observed_at, delay, predictions = _trip_update(entity, calls, vehicle.current_stop_sequence, generated_at, context)
  return TripObservation(entity_id, instance, route_id, relationship, observed_at, delay, predictions), vehicle


def _instance(descriptor: dict, source_id: str, context: str) -> TripInstance:
  trip_id = _string(descriptor, "trip_id", f"{context}.trip")
  raw_date = _string(descriptor, "start_date", f"{context}.trip", ambiguous=True)
  raw_start = _string(descriptor, "start_time", f"{context}.trip", ambiguous=True)
  if len(raw_date) != 8 or not raw_date.isdigit():
    raise RealtimeError(f"{context}.trip.start_date: invalid YYYYMMDD {raw_date!r}")
  try:
    service_date = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:]))
  except (ValueError, TypeError):
    raise RealtimeError(f"{context}.trip.start_date: invalid YYYYMMDD {raw_date!r}") from None
  return TripInstance(source_id, trip_id, service_date, _service_time(raw_start, f"{context}.trip.start_time"))


def _resolve(instance: TripInstance, descriptor: dict, schedule: Schedule, context: str) -> tuple[Trip, tuple[StopTime, ...]]:
  matches = [trip for trip in schedule.trips if trip.trip_id == instance.trip_id]
  if len(matches) != 1:
    raise RealtimeError(f"{context}: unresolved or ambiguous trip instance {instance}")
  trip = matches[0]
  route_id = _string(descriptor, "route_id", f"{context}.trip")
  if route_id != trip.route_id:
    raise RealtimeError(f"{context}.trip.route_id: expected {trip.route_id!r}, got {route_id!r}")
  if not _service_active(schedule, trip.service_id, instance.service_date):
    raise RealtimeError(f"{context}: service {trip.service_id!r} is inactive on {instance.service_date}")
  frequencies = [row for row in schedule.frequencies if row.trip_id == trip.trip_id]
  calls = dict(schedule.ordered_calls)[trip.trip_id]
  if frequencies:
    if not any(row.start <= instance.scheduled_start < row.end for row in frequencies):
      raise RealtimeError(f"{context}.trip.start_time: outside declared frequency intervals")
  elif instance.scheduled_start != calls[0].departure:
    raise RealtimeError(f"{context}.trip.start_time: does not match the scheduled first departure")
  return trip, calls


def _service_active(schedule: Schedule, service_id: str, service_date: date) -> bool:
  service = next(item for item in schedule.services if item.service_id == service_id)
  active = service.start_date <= service_date <= service.end_date and service.weekdays[service_date.weekday()]
  exception = next((item for item in schedule.exceptions if item.service_id == service_id and item.service_date == service_date), None)
  return exception.added if exception is not None else active


def _vehicle(
  entity: dict,
  entity_id: str,
  instance: TripInstance,
  calls: tuple[StopTime, ...],
  generated_at: int,
  context: str,
) -> VehicleObservation:
  value = _object(entity.get("vehicle"), f"{context}.vehicle")
  observed_at = _integer(value, "timestamp", f"{context}.vehicle", minimum=1)
  if observed_at > generated_at:
    raise RealtimeError(f"{context}.vehicle.timestamp: observation {observed_at} is newer than feed {generated_at}")
  sequence = _integer(value, "current_stop_sequence", f"{context}.vehicle", minimum=0)
  call = _call(calls, sequence, context)
  stop_id = _string(value, "current_stop_id", f"{context}.vehicle")
  if stop_id != call.stop_id:
    raise RealtimeError(f"{context}.vehicle.current_stop_id: sequence {sequence} resolves to {call.stop_id!r}, got {stop_id!r}")
  status = _string(value, "current_status", f"{context}.vehicle")
  if status not in SUPPORTED_VEHICLE_STATES:
    raise RealtimeError(f"{context}.vehicle.current_status: unsupported {status!r}")
  return VehicleObservation(entity_id, _string(value, "vehicle_id", f"{context}.vehicle"), instance, observed_at, stop_id, sequence, status)


def _trip_update(
  entity: dict,
  calls: tuple[StopTime, ...],
  current_sequence: int,
  generated_at: int,
  context: str,
) -> tuple[int, int, tuple[StopPrediction, ...]]:
  value = _object(entity.get("trip_update"), f"{context}.trip_update")
  observed_at = _integer(value, "timestamp", f"{context}.trip_update", minimum=1)
  if observed_at > generated_at:
    raise RealtimeError(f"{context}.trip_update.timestamp: observation {observed_at} is newer than feed {generated_at}")
  delay = _integer(value, "delay", f"{context}.trip_update")
  delay_sequence = _integer(value, "delay_at_stop_sequence", f"{context}.trip_update", minimum=0)
  if delay_sequence != current_sequence:
    raise RealtimeError(f"{context}.trip_update.delay: declared at stop {delay_sequence}, vehicle is at {current_sequence}")
  updates = _list(value, "stop_time_updates", f"{context}.trip_update")
  predictions = tuple(
    _prediction(_object(item, f"{context}.stop_time_updates[{index}]"), calls, current_sequence, context)
    for index, item in enumerate(updates)
  )
  sequences = [item.stop_sequence for item in predictions]
  if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
    raise RealtimeError(f"{context}.trip_update.stop_time_updates: stop sequences must be unique and ordered")
  return observed_at, delay, predictions


def _prediction(value: dict, calls: tuple[StopTime, ...], current_sequence: int, context: str) -> StopPrediction:
  sequence = _integer(value, "stop_sequence", f"{context}.stop_time_update", minimum=0)
  if sequence <= current_sequence:
    raise RealtimeError(f"{context}.stop_time_update: sequence {sequence} is not future of {current_sequence}")
  if value.get("location_group_id") or value.get("location_id"):
    raise RealtimeError(f"{context}.stop_time_update: flexible locations are unsupported")
  call = _call(calls, sequence, context)
  stop_id = _string(value, "stop_id", f"{context}.stop_time_update")
  if stop_id != call.stop_id:
    raise RealtimeError(f"{context}.stop_time_update.stop_id: sequence {sequence} resolves to {call.stop_id!r}, got {stop_id!r}")
  arrival = _event(value.get("arrival"), f"{context}.stop_time_update[{sequence}].arrival")
  departure = _event(value.get("departure"), f"{context}.stop_time_update[{sequence}].departure")
  if arrival is None and departure is None:
    raise RealtimeError(f"{context}.stop_time_update[{sequence}]: arrival or departure is required")
  if arrival is not None and departure is not None and arrival.predicted_at is not None and departure.predicted_at is not None:
    if departure.predicted_at < arrival.predicted_at:
      raise RealtimeError(f"{context}.stop_time_update[{sequence}].departure: precedes arrival")
  return StopPrediction(stop_id, sequence, arrival, departure)


def _event(value: object, context: str) -> EventPrediction | None:
  if value is None:
    return None
  event = _object(value, context)
  delay = _optional_integer(event, "delay", context)
  predicted_at = _optional_integer(event, "time", context, minimum=1)
  if delay is None and predicted_at is None:
    raise RealtimeError(f"{context}: delay or time is required")
  return EventPrediction(delay, predicted_at)


def _call(calls: tuple[StopTime, ...], sequence: int, context: str) -> StopTime:
  matches = [call for call in calls if call.stop_sequence == sequence]
  if len(matches) != 1:
    raise RealtimeError(f"{context}: unknown stop sequence {sequence}")
  return matches[0]


def _contradictions(trips: list[TripObservation], vehicles: list[VehicleObservation]) -> None:
  relationships: dict[TripInstance, set[str]] = {}
  for trip in trips:
    relationships.setdefault(trip.instance, set()).add(trip.relationship)
  for instance, values in relationships.items():
    if len(values) > 1:
      raise RealtimeError(f"trip instance {instance}: contradictory relationships {sorted(values)}")
  _duplicates("trip instance", (trip.instance for trip in trips))
  _duplicates("vehicle id", (vehicle.vehicle_id for vehicle in vehicles))


def _service_time(value: str, context: str) -> int:
  parts = value.split(":")
  if len(parts) != 3 or not all(part.isdigit() for part in parts):
    raise RealtimeError(f"{context}: invalid service-day time {value!r}")
  hour, minute, second = map(int, parts)
  if minute >= 60 or second >= 60:
    raise RealtimeError(f"{context}: invalid service-day time {value!r}")
  return hour * 3600 + minute * 60 + second


def _object(value: object, context: str) -> dict:
  if not isinstance(value, dict):
    raise RealtimeError(f"{context}: expected object")
  return value


def _list(value: dict, field: str, context: str) -> list:
  result = value.get(field)
  if not isinstance(result, list):
    raise RealtimeError(f"{context}.{field}: expected list")
  return result


def _string(value: dict, field: str, context: str, *, ambiguous: bool = False) -> str:
  result = value.get(field)
  if not isinstance(result, str) or not result:
    reason = "required to resolve an unambiguous trip instance" if ambiguous else "required string"
    raise RealtimeError(f"{context}.{field}: {reason}")
  return result


def _integer(value: dict, field: str, context: str, *, minimum: int | None = None) -> int:
  result = value.get(field)
  if isinstance(result, bool) or not isinstance(result, int):
    raise RealtimeError(f"{context}.{field}: required integer")
  if minimum is not None and result < minimum:
    raise RealtimeError(f"{context}.{field}: expected at least {minimum}, got {result}")
  return result


def _optional_integer(value: dict, field: str, context: str, *, minimum: int | None = None) -> int | None:
  if field not in value:
    return None
  return _integer(value, field, context, minimum=minimum)


def _duplicates(name: str, values) -> None:
  seen = set()
  for value in values:
    if value in seen:
      raise RealtimeError(f"{name}: duplicate {value!r}")
    seen.add(value)


if __name__ == "__main__":
  main()
