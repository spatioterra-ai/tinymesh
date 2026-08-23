"""Audit regular clocks against the frozen MBTA event task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.mbta_clock import EXPECTED_DIGESTS, INTERVALS, admissible
from experiments.tools.mbta_headway_task import _digest, open_task


REFERENCES = {
  "submodules/pytorch-geometric-temporal": "fe555bc30ee197755c4b58a89407033a5f383415",
}


class ClockBuildError(ValueError):
  """Source or frozen protocol drift prevents a clock decision."""


def build(
  source_dir: Path,
  population_audit_path: Path,
  task_protocol_path: Path,
  topology_protocol_path: Path,
) -> dict:
  """Rebuild frozen facts and measure each candidate without materializing cells."""
  digests = _verify_artifacts(
    source_dir / "manifest.json",
    population_audit_path,
    task_protocol_path,
    topology_protocol_path,
  )
  connection, rebuilt_task, _, _ = open_task(source_dir, population_audit_path)
  connection.execute("SET threads = 1")
  task = _read(task_protocol_path)
  topology = _read(topology_protocol_path)
  if rebuilt_task != task:
    raise ClockBuildError("task: frozen protocol does not match rebuilt source facts")
  if topology.get("task_protocol_sha256") != _digest(task) or topology.get("targets") != 940_551:
    raise ClockBuildError("topology: frozen protocol does not match the task")

  population = {
    "physical_departures": connection.execute("SELECT count(*) FROM events").fetchone()[0],
    "targets": connection.execute("SELECT count(*) FROM task_targets").fetchone()[0],
    "lanes": connection.execute(
      "SELECT count(DISTINCT (parent_station, trunk_route_id, direction_id)) FROM events"
    ).fetchone()[0],
  }
  candidates = [_candidate(connection, seconds) for seconds in INTERVALS]
  admitted = [
    candidate["seconds"]
    for candidate in candidates
    if admissible(candidate["identity_retained"], candidate["causal_collision_bins"])
  ]
  selected = max(admitted, default=None)
  return {
    "schema": 1,
    **digests,
    "population": population,
    "clock_domain": "inclusive_lane_day_span_from_first_event_bin_through_last_event_bin",
    "bin_semantics": "UTC_half_open_[start,end)",
    "collision": "two_or_more_distinct_lane_event_timestamps_in_one_bin",
    "target_identity": "exact_predecessor_and_target_occupy_distinct_bins",
    "candidates": candidates,
    "selected_seconds": selected,
    "decision": f"advance:snapshot_{selected}" if selected is not None else "stop:no_identity_preserving_clock",
    "stage_1_consequence": "refine:stage_1_snapshot" if selected is not None else "close:stage_1_snapshot",
    "references": REFERENCES,
  }


def _candidate(connection: Any, seconds: int) -> dict:
  table = f"clock_{seconds}"
  spans = f"clock_{seconds}_spans"
  connection.execute(
    f"""
    CREATE TEMP TABLE {table} AS
    SELECT service_date, parent_station, trunk_route_id, direction_id,
      floor(departure_timestamp / ?)::BIGINT AS bin,
      count(*) AS events, count(DISTINCT departure_timestamp) AS event_times
    FROM events
    GROUP BY ALL
    """,
    [seconds],
  )
  connection.execute(
    f"""
    CREATE TEMP TABLE {spans} AS
    SELECT service_date, parent_station, trunk_route_id, direction_id,
      min(bin) AS first_bin, max(bin) AS last_bin,
      count(*) AS occupied_cells, sum(events) AS event_records
    FROM {table}
    GROUP BY ALL
    """
  )
  cells, occupied, events = connection.execute(
    f"SELECT sum(last_bin - first_bin + 1), sum(occupied_cells), sum(event_records) FROM {spans}"
  ).fetchone()
  collision_bins, collision_events, maximum = connection.execute(
    f"""
    SELECT count(*) FILTER (WHERE event_times > 1),
      coalesce(sum(events) FILTER (WHERE event_times > 1), 0), max(events)
    FROM {table}
    """
  ).fetchone()
  equal_sets, equal_events = connection.execute(
    """
    SELECT count(*) FILTER (WHERE event_count > 1),
      coalesce(sum(event_count) FILTER (WHERE event_count > 1), 0)
    FROM lane_times
    """
  ).fetchone()
  split_targets = _target_slices(connection, seconds, "split")
  route_targets = _target_slices(connection, seconds, "route_id")
  targets = sum(row["targets"] for row in split_targets)
  retained = sum(row["retained_targets"] for row in split_targets)
  empty = cells - occupied
  return {
    "seconds": seconds,
    "cells": cells,
    "occupied_cells": occupied,
    "empty_cells": empty,
    "empty_rate": round(empty / cells, 6),
    "event_records": events,
    "cells_per_event": round(cells / events, 6),
    "causal_collision_bins": collision_bins,
    "causal_collision_events": collision_events,
    "equal_time_sets": equal_sets,
    "equal_time_events": equal_events,
    "maximum_events_per_cell": maximum,
    "targets": targets,
    "retained_targets": retained,
    "colliding_targets": targets - retained,
    "identity_retained": retained == targets,
    "split_targets": split_targets,
    "route_targets": route_targets,
    "collision_extremes": _extremes(connection, table, collisions=True),
    "work_extremes": _extremes(connection, spans, collisions=False),
  }


def _target_slices(connection: Any, seconds: int, field: str) -> list[dict[str, Any]]:
  return _rows(
    connection,
    f"""
    SELECT {field}, count(*) AS targets,
      count(*) FILTER (WHERE floor(source_timestamp / ?) != floor(departure_timestamp / ?)) AS retained_targets,
      count(*) FILTER (WHERE floor(source_timestamp / ?) = floor(departure_timestamp / ?)) AS colliding_targets
    FROM task_targets
    GROUP BY {field}
    ORDER BY {field}
    """,
    [seconds, seconds, seconds, seconds],
  )


def _extremes(connection: Any, table: str, *, collisions: bool) -> list[dict[str, Any]]:
  if collisions:
    return _rows(
      connection,
      f"""
      SELECT service_date, parent_station, trunk_route_id, direction_id,
        count(*) AS collision_bins, sum(events) AS collision_events,
        max(events) AS maximum_events_per_cell
      FROM {table}
      WHERE event_times > 1
      GROUP BY ALL
      ORDER BY collision_events DESC, collision_bins DESC, service_date,
        parent_station, trunk_route_id, direction_id
      LIMIT 10
      """,
    )
  return _rows(
    connection,
    f"""
    SELECT service_date, parent_station, trunk_route_id, direction_id,
      last_bin - first_bin + 1 AS cells, occupied_cells,
      last_bin - first_bin + 1 - occupied_cells AS empty_cells
    FROM {table}
    ORDER BY empty_cells DESC, cells DESC, service_date,
      parent_station, trunk_route_id, direction_id
    LIMIT 10
    """,
  )


def _rows(connection: Any, query: str, parameters: list[int] | None = None) -> list[dict[str, Any]]:
  cursor = connection.execute(query, parameters or [])
  columns = tuple(column[0] for column in cursor.description)
  return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _read(path: Path) -> dict:
  try:
    value = json.loads(path.read_bytes())
  except (FileNotFoundError, json.JSONDecodeError) as error:
    raise ClockBuildError(f"{path.name}: missing or invalid frozen artifact") from error
  if not isinstance(value, dict):
    raise ClockBuildError(f"{path.name}: frozen artifact is not an object")
  return value


def _verify_artifacts(manifest: Path, population: Path, task: Path, topology: Path) -> dict[str, str]:
  paths = {
    "source_manifest_sha256": manifest,
    "population_audit_sha256": population,
    "task_protocol_sha256": task,
    "topology_protocol_sha256": topology,
  }
  try:
    observed = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
  except FileNotFoundError as error:
    raise ClockBuildError(f"{error.filename}: missing frozen artifact") from error
  if observed != EXPECTED_DIGESTS:
    raise ClockBuildError("artifacts: frozen digest drift")
  return observed


def _write(path: Path, value: object) -> None:
  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--population-audit", type=Path, required=True)
  parser.add_argument("--task-protocol", type=Path, required=True)
  parser.add_argument("--topology-protocol", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  arguments = parser.parse_args()
  result = build(
    arguments.source_dir,
    arguments.population_audit,
    arguments.task_protocol,
    arguments.topology_protocol,
  )
  _write(arguments.output, result)
  print(json.dumps({"decision": result["decision"], "output": str(arguments.output)}))


if __name__ == "__main__":
  main()
