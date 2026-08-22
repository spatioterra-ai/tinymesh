"""Validate the retained MBTA population audit and its Stage 2 decision."""

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
    availability=audit["availability"],
    decision=audit["decision"],
    stage_3_consequence="blocked:retain_no_task_until_schedule_identity_is_recoverable",
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
  if audit.get("schema") != 1 or audit.get("source_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
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
  expected = "stop:insufficient_schedule_identity" if unresolved else "advance:stage_3"
  if audit.get("decision") != expected:
    raise PopulationError("audit: decision does not follow evidence")


if __name__ == "__main__":
  main()
