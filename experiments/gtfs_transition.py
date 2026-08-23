"""Evaluate pure transitions between normalized GTFS Realtime snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal

from experiments.gtfs_realtime import FIXTURE, Snapshot, TripInstance, VehicleObservation, normalize
from experiments.gtfs_schedule import load as load_schedule


BLOCKING = "blocking"
POLICY = "policy"


class TransitionError(ValueError):
  """An invalid transition-evaluation input."""


@dataclass(frozen=True)
class TransitionPolicy:
  max_feed_gap_seconds: int
  max_trip_age_seconds: int
  max_vehicle_age_seconds: int


@dataclass(frozen=True, order=True)
class VehicleKey:
  instance: TripInstance
  vehicle_id: str


@dataclass(frozen=True, kw_only=True)
class _FindingFields:
  code: str = field(init=False)
  severity: str = field(init=False)
  source_id: str
  instance: TripInstance | None = field(default=None, init=False)
  vehicle_id: str | None = field(default=None, init=False)
  previous_stop_sequence: int | None = field(default=None, init=False)
  current_stop_sequence: int | None = field(default=None, init=False)
  previous_at: int | None = field(default=None, init=False)
  current_at: int | None = field(default=None, init=False)
  as_of: int
  limit_seconds: int | None = field(default=None, init=False)


@dataclass(frozen=True, kw_only=True)
class FeedGenerationRegressed(_FindingFields):
  code: Literal["FEED_GENERATION_REGRESSED"] = field(default="FEED_GENERATION_REGRESSED", init=False)
  severity: Literal["blocking"] = field(default=BLOCKING, init=False)
  previous_at: int = field()
  current_at: int = field()


@dataclass(frozen=True, kw_only=True)
class ContentChangedWithoutGeneration(_FindingFields):
  code: Literal["CONTENT_CHANGED_WITHOUT_GENERATION"] = field(default="CONTENT_CHANGED_WITHOUT_GENERATION", init=False)
  severity: Literal["blocking"] = field(default=BLOCKING, init=False)
  previous_at: int = field()
  current_at: int = field()


@dataclass(frozen=True, kw_only=True)
class FeedGapExceeded(_FindingFields):
  code: Literal["FEED_GAP_EXCEEDED"] = field(default="FEED_GAP_EXCEEDED", init=False)
  severity: Literal["policy"] = field(default=POLICY, init=False)
  previous_at: int = field()
  current_at: int = field()
  limit_seconds: int = field()


@dataclass(frozen=True, kw_only=True)
class ActiveAfterTerminal(_FindingFields):
  code: Literal["ACTIVE_AFTER_TERMINAL"] = field(default="ACTIVE_AFTER_TERMINAL", init=False)
  severity: Literal["blocking"] = field(default=BLOCKING, init=False)
  instance: TripInstance = field()


@dataclass(frozen=True, kw_only=True)
class TripObservationStale(_FindingFields):
  code: Literal["TRIP_OBSERVATION_STALE"] = field(default="TRIP_OBSERVATION_STALE", init=False)
  severity: Literal["policy"] = field(default=POLICY, init=False)
  instance: TripInstance = field()
  current_at: int = field()
  limit_seconds: int = field()


@dataclass(frozen=True, kw_only=True)
class VehicleStopRegressed(_FindingFields):
  code: Literal["VEHICLE_STOP_REGRESSED"] = field(default="VEHICLE_STOP_REGRESSED", init=False)
  severity: Literal["blocking"] = field(default=BLOCKING, init=False)
  instance: TripInstance = field()
  vehicle_id: str = field()
  previous_stop_sequence: int = field()
  current_stop_sequence: int = field()


@dataclass(frozen=True, kw_only=True)
class VehicleObservationStale(_FindingFields):
  code: Literal["VEHICLE_OBSERVATION_STALE"] = field(default="VEHICLE_OBSERVATION_STALE", init=False)
  severity: Literal["policy"] = field(default=POLICY, init=False)
  instance: TripInstance = field()
  vehicle_id: str = field()
  current_at: int = field()
  limit_seconds: int = field()


Finding = (
  FeedGenerationRegressed
  | ContentChangedWithoutGeneration
  | FeedGapExceeded
  | ActiveAfterTerminal
  | TripObservationStale
  | VehicleStopRegressed
  | VehicleObservationStale
)


@dataclass(frozen=True)
class Transition:
  previous_fingerprint: str
  current_fingerprint: str
  findings: tuple[Finding, ...]
  eligible_trips: tuple[TripInstance, ...]
  eligible_vehicles: tuple[VehicleKey, ...]


@dataclass(frozen=True)
class Observation:
  previous_fingerprint: str
  current_fingerprint: str
  findings: tuple[str, ...]
  eligible_trips: int
  eligible_vehicles: int


def fingerprint(snapshot: Snapshot) -> str:
  """Hash normalized feed semantics independently of entity tuple order."""
  semantic = {
    "source_id": snapshot.source_id,
    "generated_at": snapshot.generated_at,
    "trips": [_plain(value) for value in sorted(snapshot.trips, key=lambda item: (item.instance, item.entity_id))],
    "vehicles": [_plain(value) for value in sorted(snapshot.vehicles, key=lambda item: (item.instance, item.vehicle_id))],
  }
  encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()


def evaluate(
  previous: Snapshot,
  current: Snapshot,
  as_of: int,
  policy: TransitionPolicy,
) -> Transition:
  """Return deterministic findings and current facts eligible for projection."""
  _validate_inputs(previous, current, as_of, policy)
  previous_fingerprint = fingerprint(previous)
  current_fingerprint = fingerprint(current)
  findings: list[Finding] = []

  if current.generated_at < previous.generated_at:
    findings.append(
      FeedGenerationRegressed(
        source_id=current.source_id, as_of=as_of, previous_at=previous.generated_at, current_at=current.generated_at
      )
    )
  elif current.generated_at == previous.generated_at and current_fingerprint != previous_fingerprint:
    findings.append(
      ContentChangedWithoutGeneration(
        source_id=current.source_id,
        as_of=as_of,
        previous_at=previous.generated_at,
        current_at=current.generated_at,
      )
    )

  gap = current.generated_at - previous.generated_at
  if gap > policy.max_feed_gap_seconds:
    findings.append(
      FeedGapExceeded(
        source_id=current.source_id,
        as_of=as_of,
        previous_at=previous.generated_at,
        current_at=current.generated_at,
        limit_seconds=policy.max_feed_gap_seconds,
      )
    )

  previous_trips = {trip.instance: trip for trip in previous.trips}
  previous_vehicles = {_vehicle_key(vehicle): vehicle for vehicle in previous.vehicles}
  for trip in current.trips:
    prior = previous_trips.get(trip.instance)
    if prior is not None and prior.relationship == "CANCELED" and trip.relationship != "CANCELED":
      findings.append(ActiveAfterTerminal(source_id=current.source_id, as_of=as_of, instance=trip.instance))
    if as_of - trip.observed_at > policy.max_trip_age_seconds:
      findings.append(
        TripObservationStale(
          source_id=current.source_id,
          as_of=as_of,
          instance=trip.instance,
          current_at=trip.observed_at,
          limit_seconds=policy.max_trip_age_seconds,
        )
      )
  for vehicle in current.vehicles:
    key = _vehicle_key(vehicle)
    prior = previous_vehicles.get(key)
    if prior is not None and vehicle.current_stop_sequence < prior.current_stop_sequence:
      findings.append(
        VehicleStopRegressed(
          source_id=current.source_id,
          as_of=as_of,
          instance=vehicle.instance,
          vehicle_id=vehicle.vehicle_id,
          previous_stop_sequence=prior.current_stop_sequence,
          current_stop_sequence=vehicle.current_stop_sequence,
        )
      )
    if as_of - vehicle.observed_at > policy.max_vehicle_age_seconds:
      findings.append(
        VehicleObservationStale(
          source_id=current.source_id,
          as_of=as_of,
          instance=vehicle.instance,
          vehicle_id=vehicle.vehicle_id,
          current_at=vehicle.observed_at,
          limit_seconds=policy.max_vehicle_age_seconds,
        )
      )

  ordered = tuple(sorted(findings, key=_finding_key))
  eligible_trips, eligible_vehicles = _eligibility(current, ordered)
  return Transition(previous_fingerprint, current_fingerprint, ordered, eligible_trips, eligible_vehicles)


def observe(path: str | Path = FIXTURE) -> Observation:
  schedule = load_schedule()
  current = normalize(json.loads(Path(path).read_text()), schedule, schedule.source_id)
  prior_trips = tuple(replace(trip, observed_at=trip.observed_at - 20) for trip in current.trips)
  prior_vehicle = replace(
    current.vehicles[0],
    observed_at=current.vehicles[0].observed_at - 20,
    current_stop_id="STAGECOACH",
    current_stop_sequence=1,
  )
  previous = replace(current, generated_at=current.generated_at - 20, trips=prior_trips, vehicles=(prior_vehicle,))
  transition = evaluate(previous, current, current.generated_at, TransitionPolicy(30, 90, 90))
  return Observation(
    transition.previous_fingerprint,
    transition.current_fingerprint,
    tuple(finding.code for finding in transition.findings),
    len(transition.eligible_trips),
    len(transition.eligible_vehicles),
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _validate_inputs(previous: Snapshot, current: Snapshot, as_of: int, policy: TransitionPolicy) -> None:
  if previous.source_id != current.source_id:
    raise TransitionError(f"source mismatch: {previous.source_id!r} != {current.source_id!r}")
  if as_of < current.generated_at:
    raise TransitionError(f"as_of {as_of} precedes current generation {current.generated_at}")
  limits = asdict(policy)
  for name, value in limits.items():
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise TransitionError(f"policy.{name}: expected non-negative integer")


def _eligibility(snapshot: Snapshot, findings: tuple[Finding, ...]) -> tuple[tuple[TripInstance, ...], tuple[VehicleKey, ...]]:
  trips = {trip.instance for trip in snapshot.trips}
  vehicles = {_vehicle_key(vehicle) for vehicle in snapshot.vehicles}
  if any(isinstance(finding, (FeedGenerationRegressed, ContentChangedWithoutGeneration)) for finding in findings):
    return (), ()
  for finding in findings:
    if isinstance(finding, (ActiveAfterTerminal, TripObservationStale)):
      trips.discard(finding.instance)
      if isinstance(finding, ActiveAfterTerminal):
        vehicles = {vehicle for vehicle in vehicles if vehicle.instance != finding.instance}
    if isinstance(finding, (VehicleStopRegressed, VehicleObservationStale)):
      vehicles.discard(VehicleKey(finding.instance, finding.vehicle_id))
  return tuple(sorted(trips)), tuple(sorted(vehicles))


def _vehicle_key(vehicle: VehicleObservation) -> VehicleKey:
  return VehicleKey(vehicle.instance, vehicle.vehicle_id)


def _finding_key(finding: Finding) -> tuple:
  return (finding.code, repr(finding.instance), finding.vehicle_id or "")


def _plain(value):
  if is_dataclass(value):
    return _plain(asdict(value))
  if isinstance(value, dict):
    return {key: _plain(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_plain(item) for item in value]
  if isinstance(value, date):
    return value.isoformat()
  return value


if __name__ == "__main__":
  main()
