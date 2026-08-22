"""Validate the retained MBTA population audit and its Stage 3 boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "mbta_population"


class PopulationError(ValueError):
  """The retained population record violates its evidence contract."""


@dataclass(frozen=True)
class Observation:
  source_files: int
  source_bytes: int
  cap_bytes: int
  dates: int
  routes: int
  source_rows: int
  trip_instances: int
  missing_movement: int
  source_headways: int
  schedule_resolved: int
  schedule_unresolved: int
  unresolved_added: int
  unresolved_nonrevenue: int
  unresolved_other: int
  schedule_versions: int
  represented_source_rows: int
  physical_departures: int
  duplicate_aliases: int
  conflicting_aliases: int
  ambiguous_order_trips: int
  ambiguous_order_rows: int
  run_relations: int
  ambiguous_run_sources: int
  ambiguous_run_targets: int
  headway_relations: int
  exact_headways: int
  mismatched_headways: int
  boundary_only_headways: int
  simultaneous_events: int
  station_directions: int
  median_gap_seconds: int
  p95_gap_seconds: int
  maximum_gap_seconds: int
  availability: str
  decision: str
  stage_3_consequence: str


def observe(path: str | Path = FIXTURE) -> Observation:
  directory = Path(path)
  manifest_bytes = (directory / "manifest.json").read_bytes()
  manifest = json.loads(manifest_bytes)
  audit = json.loads((directory / "audit.json").read_bytes())
  _validate_manifest(manifest)
  _validate_audit(audit, manifest, manifest_bytes)
  population = audit["population"]
  events = audit["event_population"]
  versions = {row["feed_version"] for row in audit["schedule_versions"]}
  return Observation(
    source_files=audit["source_files"],
    source_bytes=audit["source_bytes"],
    cap_bytes=manifest["cap_bytes"],
    dates=population["dates"],
    routes=len(population["routes"]),
    source_rows=population["source_rows"],
    trip_instances=population["trip_instances"],
    missing_movement=population["missing_movement"],
    source_headways=population["source_headways"],
    schedule_resolved=population["schedule_resolved"],
    schedule_unresolved=population["schedule_unresolved"],
    unresolved_added=population["unresolved_added"],
    unresolved_nonrevenue=population["unresolved_nonrevenue"],
    unresolved_other=population["unresolved_other"],
    schedule_versions=len(versions),
    represented_source_rows=events["represented_source_rows"],
    physical_departures=events["physical_departures"],
    duplicate_aliases=events["duplicate_aliases"],
    conflicting_aliases=events["conflicting_aliases"],
    ambiguous_order_trips=events["ambiguous_order_trips"],
    ambiguous_order_rows=events["ambiguous_order_rows"],
    run_relations=events["run_relations"],
    ambiguous_run_sources=events["ambiguous_run_sources"],
    ambiguous_run_targets=events["ambiguous_run_targets"],
    headway_relations=events["headway_relations"],
    exact_headways=events["exact_headways"],
    mismatched_headways=events["mismatched_headways"],
    boundary_only_headways=events["boundary_only_headways"],
    simultaneous_events=events["simultaneous_events"],
    station_directions=events["station_directions"],
    median_gap_seconds=events["median_gap_seconds"],
    p95_gap_seconds=events["p95_gap_seconds"],
    maximum_gap_seconds=events["maximum_gap_seconds"],
    availability=audit["availability"],
    decision=audit["decision"],
    stage_3_consequence="advance:retrospective_event_time_with_schedule_mask",
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _validate_manifest(manifest: dict) -> None:
  sources = manifest.get("sources", ())
  if manifest.get("schema") != 1 or manifest.get("start_date") != "2026-07-24" or manifest.get("end_date") != "2026-08-20":
    raise PopulationError("manifest: unsupported population")
  if manifest.get("availability") != "retrospective_event_time_only:no_generation_or_ingestion_clock":
    raise PopulationError("manifest: unsupported availability claim")
  if len(sources) != 34 or len({source["filename"] for source in sources}) != len(sources):
    raise PopulationError("manifest: source inventory drift")
  if any(source["bytes"] <= 0 or len(source.get("sha256") or "") != 64 for source in sources):
    raise PopulationError("manifest: unsealed source")
  total = sum(source["bytes"] for source in sources)
  if total != 110_610_188 or total > manifest.get("cap_bytes", 0):
    raise PopulationError("manifest: source byte drift")


def _validate_audit(audit: dict, manifest: dict, manifest_bytes: bytes) -> None:
  if audit.get("schema") != 2 or audit.get("source_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
    raise PopulationError("audit: manifest drift")
  if (audit.get("source_files"), audit.get("source_bytes")) != (len(manifest["sources"]), 110_610_188):
    raise PopulationError("audit: source boundary drift")
  groups = audit.get("date_routes", ())
  identities = {(row["service_date"], row["route_id"]) for row in groups}
  population = audit.get("population", {})
  dates = {row["service_date"] for row in groups}
  routes = {row["route_id"] for row in groups}
  if len(groups) != len(identities) or (len(dates), len(routes), len(groups)) != (28, 8, 224):
    raise PopulationError("audit: date-route coverage drift")
  summed = (
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
  if any(sum(row[field] for row in groups) != population.get(field) for field in summed):
    raise PopulationError("audit: population total drift")
  unresolved = population.get("schedule_unresolved")
  if unresolved != sum(population.get(field, 0) for field in ("unresolved_added", "unresolved_nonrevenue", "unresolved_other")):
    raise PopulationError("audit: unresolved classification drift")
  if population.get("schedule_resolved", 0) + unresolved != population.get("source_rows"):
    raise PopulationError("audit: Schedule coverage drift")
  versions = audit.get("schedule_versions", ())
  if len(versions) != 28 or {row["service_date"] for row in versions} != dates:
    raise PopulationError("audit: Schedule version drift")
  events = audit.get("event_population", {})
  expected_events = {
    "ambiguous_order_rows": 384,
    "ambiguous_order_trips": 39,
    "ambiguous_run_sources": 84,
    "ambiguous_run_targets": 56,
    "boundary_only_headways": 1_007,
    "conflicting_aliases": 0,
    "derived_only_headways": 0,
    "duplicate_aliases": 225,
    "exact_headways": 940_776,
    "headway_relations": 940_752,
    "maximum_gap_seconds": 57_860,
    "median_gap_seconds": 399,
    "minimum_gap_seconds": 1,
    "mismatched_headways": 201,
    "p95_gap_seconds": 960,
    "physical_departures": 947_489,
    "represented_source_rows": 947_714,
    "run_relations": 877_168,
    "simultaneous_events": 413,
    "simultaneous_groups": 206,
    "station_directions": 259,
  }
  if events != expected_events:
    raise PopulationError("audit: event population drift")
  if events["represented_source_rows"] != events["physical_departures"] + events["duplicate_aliases"]:
    raise PopulationError("audit: physical identity drift")
  if events["exact_headways"] + events["mismatched_headways"] + events["boundary_only_headways"] != population.get(
    "source_headways"
  ):
    raise PopulationError("audit: headway coverage drift")
  event_sums = {
    field: sum(row[field] for row in groups)
    for field in (
      "ambiguous_order_rows",
      "ambiguous_order_trips",
      "boundary_only_headways",
      "exact_headways",
      "mismatched_headways",
      "represented_source_rows",
      "run_relations",
    )
  }
  if any(event_sums[field] != events[field] for field in event_sums):
    raise PopulationError("audit: event group drift")
  if any(
    row["exact_headways"] + row["mismatched_headways"] + row["boundary_only_headways"] != row["source_headways"]
    for row in groups
  ):
    raise PopulationError("audit: group headway coverage drift")
  sufficient = events["exact_headways"] > 0 and events["conflicting_aliases"] == 0
  expected = (
    "advance:stage_3_retrospective_with_schedule_mask"
    if sufficient and unresolved
    else "advance:stage_3_retrospective"
    if sufficient
    else "stop:insufficient_event_population"
  )
  if audit.get("decision") != expected:
    raise PopulationError("audit: decision does not follow evidence")


if __name__ == "__main__":
  main()
