"""Validate the frozen MBTA next-headway task and its causal boundary."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal


FIXTURE = Path(__file__).parent / "fixtures" / "mbta_headway_task"
POPULATION_AUDIT = Path(__file__).parent / "fixtures" / "mbta_population" / "audit.json"
Split = Literal["train", "validation", "test"]


class TaskError(ValueError):
  """The task or retained evidence violates its frozen contract."""


@dataclass(frozen=True, order=True)
class Lane:
  parent_station: str
  trunk_route_id: str
  direction_id: int


@dataclass(frozen=True, order=True)
class Event:
  service_date: date
  timestamp: int
  vehicle_id: str
  lane: Lane


@dataclass(frozen=True)
class HeadwayTarget:
  source: Event
  target: Event
  cutoff: int
  seconds: int


@dataclass(frozen=True)
class Observation:
  availability: str
  targets: int
  train_targets: int
  validation_targets: int
  test_targets: int
  split: Split
  temporal_bin_hours: int
  temporal_minimum_support: int
  persistence_mae_seconds: float
  temporal_mae_seconds: float
  plan_mae_seconds: float
  best_baseline: str
  decision: str


def strict_target(events: tuple[Event, ...], source: Event, target: Event) -> HeadwayTarget:
  """Freeze the next strictly later event time in one service-day lane."""
  if len(events) != len(set(events)) or source not in events or target not in events:
    raise TaskError("target: events must have unique physical identities and contain both endpoints")
  if (source.service_date, source.lane) != (target.service_date, target.lane):
    raise TaskError("target: endpoints must share one service-day lane")
  lane = (source.service_date, source.lane)
  later = [
    event.timestamp
    for event in events
    if (event.service_date, event.lane) == lane and event.timestamp > source.timestamp
  ]
  if not later or target.timestamp != min(later):
    raise TaskError("target: endpoint is not the next strict departure time")
  return HeadwayTarget(source, target, source.timestamp + 1, target.timestamp - source.timestamp)


def causal_prefix(events: tuple[Event, ...], target: HeadwayTarget) -> tuple[Event, ...]:
  """Return exactly the deterministic event prefix visible at the task cutoff."""
  return tuple(sorted(event for event in events if event.timestamp < target.cutoff))


def split_for(service_date: date) -> Split:
  if date(2026, 7, 24) <= service_date <= date(2026, 8, 10):
    return "train"
  if date(2026, 8, 11) <= service_date <= date(2026, 8, 15):
    return "validation"
  if date(2026, 8, 16) <= service_date <= date(2026, 8, 20):
    return "test"
  raise TaskError(f"split: date outside frozen population: {service_date}")


def observe(path: str | Path = FIXTURE, *, test: bool = False) -> Observation:
  directory = Path(path)
  protocol = _read(directory / "protocol.json")
  validation = _read(directory / "validation.json")
  _validate_protocol(protocol)
  _validate_evidence(validation, "validation", protocol)
  evidence = validation
  if test:
    evidence = _read(directory / "test.json")
    _validate_evidence(evidence, "test", protocol)
    if evidence.get("validation_sha256") != _digest(validation):
      raise TaskError("test: validation artifact drift")
  counts = {row["split"]: row["targets"] for row in protocol["target_counts"]}
  metrics = {row["baseline"]: row for row in evidence["metrics"]}
  best = min(metrics.values(), key=lambda row: (row["mae_seconds"], row["baseline"]))
  selected = evidence["selected_temporal"]
  return Observation(
    availability=protocol["availability"],
    targets=sum(counts.values()),
    train_targets=counts["train"],
    validation_targets=counts["validation"],
    test_targets=counts["test"],
    split=evidence["split"],
    temporal_bin_hours=selected["bin_hours"],
    temporal_minimum_support=selected["minimum_support"],
    persistence_mae_seconds=metrics["persistence"]["mae_seconds"],
    temporal_mae_seconds=metrics["temporal"]["mae_seconds"],
    plan_mae_seconds=metrics["plan"]["mae_seconds"],
    best_baseline=best["baseline"],
    decision=evidence["decision"],
  )


def _validate_protocol(protocol: dict) -> None:
  expected_splits = {
    "train": ["2026-07-24", "2026-08-10"],
    "validation": ["2026-08-11", "2026-08-15"],
    "test": ["2026-08-16", "2026-08-20"],
  }
  counts = protocol.get("target_counts", ())
  totals = {row.get("split"): row.get("targets") for row in counts}
  if protocol.get("schema") != 1 or protocol.get("splits") != expected_splits:
    raise TaskError("protocol: unsupported task or temporal split")
  if totals != {"train": 625_073, "validation": 138_910, "test": 176_568}:
    raise TaskError("protocol: target population drift")
  if protocol.get("population_audit_sha256") != hashlib.sha256(POPULATION_AUDIT.read_bytes()).hexdigest():
    raise TaskError("protocol: population audit drift")
  if protocol.get("cutoff") != "previous_departure_timestamp_plus_one_second":
    raise TaskError("protocol: causal cutoff drift")
  if protocol.get("target") != "strict_next_movement_headway_seconds":
    raise TaskError("protocol: target drift")


def _validate_evidence(evidence: dict, split: Split, protocol: dict) -> None:
  metrics = evidence.get("metrics", ())
  names = {row.get("baseline") for row in metrics}
  if evidence.get("schema") != 1 or evidence.get("split") != split:
    raise TaskError(f"{split}: unsupported evidence")
  if evidence.get("protocol_sha256") != _digest(protocol):
    raise TaskError(f"{split}: protocol artifact drift")
  if names != {"persistence", "temporal", "plan"} or any(row.get("targets", 0) <= 0 for row in metrics):
    raise TaskError(f"{split}: baseline evidence drift")
  if any(not 0 <= row.get("coverage", -1) <= 1 for row in metrics):
    raise TaskError(f"{split}: invalid coverage")


def _read(path: Path) -> dict:
  try:
    value = json.loads(path.read_bytes())
  except (FileNotFoundError, json.JSONDecodeError) as error:
    raise TaskError(f"{path.name}: missing or invalid retained artifact") from error
  if not isinstance(value, dict):
    raise TaskError(f"{path.name}: retained artifact is not an object")
  return value


def _digest(value: object) -> str:
  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  return hashlib.sha256(encoded.encode()).hexdigest()


def main() -> None:
  print(json.dumps(asdict(observe(test=os.environ.get("TEST") == "1")), indent=2))


if __name__ == "__main__":
  main()
