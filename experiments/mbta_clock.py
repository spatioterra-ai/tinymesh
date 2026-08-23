"""Validate the retained full-population MBTA clock decision."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "mbta_clock"
INTERVALS = (30, 60, 300)


class ClockError(ValueError):
  """A clock projection or retained audit violates its declared boundary."""


@dataclass(frozen=True)
class LaneAudit:
  seconds: int
  event_records: int
  cells: int
  occupied_cells: int
  empty_cells: int
  causal_collision_bins: int
  causal_collision_events: int
  equal_time_sets: int
  equal_time_events: int
  maximum_events_per_cell: int


@dataclass(frozen=True)
class Candidate:
  seconds: int
  cells: int
  occupied_cells: int
  empty_cells: int
  empty_rate: float
  event_records: int
  cells_per_event: float
  causal_collision_bins: int
  causal_collision_events: int
  equal_time_sets: int
  equal_time_events: int
  maximum_events_per_cell: int
  targets: int
  retained_targets: int
  colliding_targets: int
  identity_retained: bool


EXPECTED_DIGESTS = {
  "source_manifest_sha256": "780076ebf32ceb735b334b5bbdba0297d809088f0689fac37067db46fd36d701",
  "population_audit_sha256": "0345659ab2aa2a4c3341954dcfd789110e9567097f69532d25f895e7fa16e1bb",
  "task_protocol_sha256": "83e048c9f6e1f7f5de1ecb70e9a0a5cee64ac699b3ce3da94f4a9cd8f60e69ed",
  "topology_protocol_sha256": "e3c44f17757820eea46c7c6e066fe914991c920388e1a56a4e17f7c013303db8",
}
EXPECTED_CANDIDATES = (
  Candidate(30, 14_333_994, 945_817, 13_388_177, 0.934016, 947_489, 15.128401,
            1_461, 2_932, 206, 413, 3, 940_551, 939_092, 1_459, False),
  Candidate(60, 7_170_046, 943_452, 6_226_594, 0.868418, 947_489, 7.567419,
            3_811, 7_648, 206, 413, 3, 940_551, 936_728, 3_823, False),
  Candidate(300, 1_438_867, 826_474, 612_393, 0.425608, 947_489, 1.518611,
            102_906, 223_764, 206, 413, 6, 940_551, 819_799, 120_752, False),
)


@dataclass(frozen=True)
class Observation:
  physical_departures: int
  targets: int
  lanes: int
  candidates: tuple[Candidate, ...]
  selected_seconds: int | None
  decision: str
  stage_1_consequence: str


def bin_of(timestamp: int, seconds: int) -> int:
  """Return the half-open UTC clock bin containing ``timestamp``."""
  if seconds <= 0:
    raise ClockError("clock: seconds must be positive")
  return timestamp // seconds


def audit_lane(timestamps: tuple[int, ...], seconds: int) -> LaneAudit:
  """Measure one lane-day without inventing cells outside its active span."""
  if not timestamps:
    raise ClockError("clock: lane-day has no events")
  bins: dict[int, list[int]] = defaultdict(list)
  for timestamp in timestamps:
    bins[bin_of(timestamp, seconds)].append(timestamp)
  first, last = min(bins), max(bins)
  collisions = tuple(values for values in bins.values() if len(set(values)) > 1)
  equal = tuple(count for count in Counter(timestamps).values() if count > 1)
  return LaneAudit(
    seconds=seconds,
    event_records=len(timestamps),
    cells=last - first + 1,
    occupied_cells=len(bins),
    empty_cells=last - first + 1 - len(bins),
    causal_collision_bins=len(collisions),
    causal_collision_events=sum(map(len, collisions)),
    equal_time_sets=len(equal),
    equal_time_events=sum(equal),
    maximum_events_per_cell=max(map(len, bins.values())),
  )


def retains_target(source_timestamp: int, target_timestamp: int, seconds: int) -> bool:
  """Return whether one snapshot boundary separates a strict target pair."""
  if source_timestamp >= target_timestamp:
    raise ClockError("target: time did not advance")
  return bin_of(source_timestamp, seconds) != bin_of(target_timestamp, seconds)


def admissible(identity_retained: bool, causal_collision_bins: int) -> bool:
  """Return whether a projection preserves the event task exactly."""
  return identity_retained and causal_collision_bins == 0


def observe(path: str | Path = FIXTURE) -> Observation:
  audit = json.loads((Path(path) / "audit.json").read_bytes())
  _validate(audit)
  population = audit["population"]
  candidates = tuple(_summary(candidate) for candidate in audit["candidates"])
  return Observation(
    physical_departures=population["physical_departures"],
    targets=population["targets"],
    lanes=population["lanes"],
    candidates=candidates,
    selected_seconds=audit["selected_seconds"],
    decision=audit["decision"],
    stage_1_consequence=audit["stage_1_consequence"],
  )


def _validate(audit: dict) -> None:
  if audit.get("schema") != 1:
    raise ClockError("audit: unsupported schema")
  if tuple(candidate.get("seconds") for candidate in audit.get("candidates", ())) != INTERVALS:
    raise ClockError("audit: candidate intervals drift")
  if any(audit.get(name) != digest for name, digest in EXPECTED_DIGESTS.items()):
    raise ClockError("audit: artifact digest drift")

  population = audit.get("population", {})
  if population != {"physical_departures": 947_489, "targets": 940_551, "lanes": 259}:
    raise ClockError("audit: frozen population drift")
  for candidate in audit["candidates"]:
    if candidate["cells"] != candidate["occupied_cells"] + candidate["empty_cells"]:
      raise ClockError("audit: cell accounting drift")
    if candidate["targets"] != candidate["retained_targets"] + candidate["colliding_targets"]:
      raise ClockError("audit: target accounting drift")
    if candidate["identity_retained"] != (candidate["colliding_targets"] == 0):
      raise ClockError("audit: identity decision drift")
    if sum(row["targets"] for row in candidate["split_targets"]) != population["targets"]:
      raise ClockError("audit: split target drift")
    if sum(row["targets"] for row in candidate["route_targets"]) != population["targets"]:
      raise ClockError("audit: route target drift")
    if len(candidate["collision_extremes"]) != 10 or len(candidate["work_extremes"]) != 10:
      raise ClockError("audit: extreme slice drift")

  if tuple(_summary(candidate) for candidate in audit["candidates"]) != EXPECTED_CANDIDATES:
    raise ClockError("audit: candidate observation drift")

  admitted = [
    candidate["seconds"]
    for candidate in audit["candidates"]
    if admissible(candidate["identity_retained"], candidate["causal_collision_bins"])
  ]
  selected = max(admitted, default=None)
  decision = f"advance:snapshot_{selected}" if selected is not None else "stop:no_identity_preserving_clock"
  if (audit.get("selected_seconds"), audit.get("decision")) != (selected, decision):
    raise ClockError("audit: selection drift")
  consequence = "refine:stage_1_snapshot" if selected is not None else "close:stage_1_snapshot"
  if audit.get("stage_1_consequence") != consequence:
    raise ClockError("audit: Stage 1 consequence drift")


def _summary(candidate: dict) -> Candidate:
  return Candidate(**{
    key: value
    for key, value in candidate.items()
    if key not in ("split_targets", "route_targets", "collision_extremes", "work_extremes")
  })


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


if __name__ == "__main__":
  main()
