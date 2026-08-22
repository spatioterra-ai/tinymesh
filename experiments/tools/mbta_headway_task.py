"""Build the frozen retrospective MBTA next-headway task and baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from experiments.tools.mbta_population import Plan, audit_events, open_population


TEMPORAL_CANDIDATES = tuple((hours, support) for hours in (1, 2, 4) for support in (4, 16, 64))
REFERENCES = {
  "submodules/libcity": "5a6391d41944e937f2c15e9be85ab7f40ac8b23e",
  "submodules/pytorch-geometric-temporal": "fe555bc30ee197755c4b58a89407033a5f383415",
  "submodules/torch-spatiotemporal": "aa5f313e000d192bdec270748b8d01df5912e58e",
}


class TaskError(ValueError):
  """The retained task or baseline evidence violates its frozen contract."""


def build(
  source_dir: Path,
  population_audit: Path,
  include_test: bool,
  frozen_protocol: dict | None = None,
  frozen_validation: dict | None = None,
) -> tuple[dict, dict, dict | None]:
  connection, protocol, selected, candidates = open_task(source_dir, population_audit)
  validation = _evidence(connection, "validation", selected, candidates)
  validation["protocol_sha256"] = _digest(protocol)
  if include_test and (protocol != frozen_protocol or validation != frozen_validation):
    raise TaskError("test: frozen validation artifacts do not match the rebuilt task")
  test = _evidence(connection, "test", selected, None) if include_test else None
  if test is not None:
    test["protocol_sha256"] = _digest(protocol)
    test["validation_sha256"] = _digest(validation)
  return protocol, validation, test


def open_task(source_dir: Path, population_audit: Path) -> tuple[Any, dict, dict, list[dict]]:
  """Open the verified population and reconstruct the frozen task in DuckDB."""
  connection, source = open_population(source_dir)
  population, _ = audit_events(connection)
  _service_days(connection)
  _targets(connection)
  _schedule(connection, source_dir, source)
  protocol = _protocol(connection, source_dir, population_audit, population)
  candidates = [_temporal_candidate(connection, hours, support) for hours, support in TEMPORAL_CANDIDATES]
  selected = min(candidates, key=lambda row: (row["mae_seconds"], row["bin_hours"], row["minimum_support"]))
  return connection, protocol, selected, candidates


def _service_days(connection: Any) -> None:
  timezone = ZoneInfo("America/New_York")
  rows = []
  day = date(2026, 7, 24)
  while day <= date(2026, 8, 20):
    rows.append((int(day.strftime("%Y%m%d")), int(datetime.combine(day, time(), timezone).timestamp()), day.weekday()))
    day = day.fromordinal(day.toordinal() + 1)
  connection.execute("CREATE TEMP TABLE service_days(service_date INTEGER, start_timestamp BIGINT, weekday INTEGER)")
  connection.executemany("INSERT INTO service_days VALUES (?, ?, ?)", rows)


def _targets(connection: Any) -> None:
  connection.execute(
    """
    CREATE TEMP TABLE event_routes AS
    SELECT service_date, vehicle_id, parent_station, direction_id, departure_timestamp,
      min(route_id) AS route_id,
      count(DISTINCT route_id) AS route_aliases,
      bool_or(schedule_resolved) AS schedule_resolved
    FROM event_aliases
    GROUP BY service_date, vehicle_id, parent_station, direction_id, departure_timestamp
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE task_targets AS
    WITH exact AS (
      SELECT relations.service_date, target.vehicle_id, relations.parent_station,
        relations.trunk_route_id, relations.direction_id, relations.departure_timestamp,
        relations.source_timestamp, relations.elapsed_seconds
      FROM headway_relations relations
      JOIN events target
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
      WHERE target.source_headway_seconds = relations.elapsed_seconds
    ), history AS (
      SELECT relations.service_date, relations.parent_station, relations.trunk_route_id,
        relations.direction_id, relations.departure_timestamp, relations.elapsed_seconds
      FROM headway_relations relations
      JOIN events target
        USING (service_date, parent_station, trunk_route_id, direction_id, departure_timestamp)
      WHERE target.source_headway_seconds = relations.elapsed_seconds
    )
    SELECT exact.*, routes.route_id, routes.route_aliases, routes.schedule_resolved,
      exact.source_timestamp + 1 AS cutoff_timestamp,
      history.elapsed_seconds AS persistence_seconds,
      CASE
        WHEN exact.service_date <= 20260810 THEN 'train'
        WHEN exact.service_date <= 20260815 THEN 'validation'
        ELSE 'test'
      END AS split,
      ((days.weekday + floor((exact.source_timestamp - days.start_timestamp) / 86400))::INT % 7) AS weekday,
      (floor((exact.source_timestamp - days.start_timestamp) / 3600)::INT % 24) AS local_hour,
      EXISTS (
        SELECT 1 FROM run_conflict_sources conflicts
        WHERE conflicts.service_date = exact.service_date
          AND conflicts.vehicle_id = exact.vehicle_id
          AND conflicts.parent_station = exact.parent_station
          AND conflicts.direction_id = exact.direction_id
          AND conflicts.departure_timestamp = exact.departure_timestamp
      ) AS ambiguous_run_source,
      EXISTS (
        SELECT 1 FROM run_conflict_targets conflicts
        WHERE conflicts.service_date = exact.service_date
          AND conflicts.target_vehicle_id = exact.vehicle_id
          AND conflicts.target_parent_station = exact.parent_station
          AND conflicts.target_direction_id = exact.direction_id
          AND conflicts.target_timestamp = exact.departure_timestamp
      ) AS ambiguous_run_target
    FROM exact
    JOIN event_routes routes
      USING (service_date, vehicle_id, parent_station, direction_id, departure_timestamp)
    JOIN service_days days USING (service_date)
    LEFT JOIN history
      ON history.service_date = exact.service_date
      AND history.parent_station = exact.parent_station
      AND history.trunk_route_id = exact.trunk_route_id
      AND history.direction_id = exact.direction_id
      AND history.departure_timestamp = exact.source_timestamp
    """
  )
  duplicate = connection.execute(
    """
    SELECT count(*) - count(DISTINCT (
      service_date, vehicle_id, parent_station, direction_id, departure_timestamp
    )) FROM task_targets
    """
  ).fetchone()[0]
  if duplicate:
    raise TaskError(f"target: {duplicate} duplicate physical identities")


def _schedule(connection: Any, source_dir: Path, source: Plan) -> None:
  trips = str(_source(source_dir, source, "trips"))
  calls = str(_source(source_dir, source, "stop_times"))
  stops = str(_source(source_dir, source, "stops"))
  calendar = str(_source(source_dir, source, "calendar"))
  exceptions = str(_source(source_dir, source, "calendar_dates"))
  connection.execute(
    """
    CREATE TEMP TABLE route_mapping AS
    SELECT route_id, min(trunk_route_id) AS trunk_route_id
    FROM population
    GROUP BY route_id
    HAVING count(DISTINCT trunk_route_id) = 1
    """
  )
  mappings = connection.execute("SELECT count(*) FROM route_mapping").fetchone()[0]
  routes = connection.execute("SELECT count(DISTINCT route_id) FROM population").fetchone()[0]
  if mappings != routes:
    raise TaskError("schedule: route-to-trunk mapping is not unique")
  connection.execute(
    """
    CREATE TEMP TABLE active_services AS
    WITH regular AS (
      SELECT DISTINCT days.service_date, calendar.service_id
      FROM service_days days
      JOIN read_parquet(?) calendar
        ON calendar.gtfs_active_date <= days.service_date AND calendar.gtfs_end_date >= days.service_date
        AND days.service_date BETWEEN calendar.start_date AND calendar.end_date
      WHERE CASE days.weekday
        WHEN 0 THEN calendar.monday WHEN 1 THEN calendar.tuesday WHEN 2 THEN calendar.wednesday
        WHEN 3 THEN calendar.thursday WHEN 4 THEN calendar.friday WHEN 5 THEN calendar.saturday
        ELSE calendar.sunday END = 1
    ), added AS (
      SELECT DISTINCT days.service_date, exceptions.service_id
      FROM service_days days
      JOIN read_parquet(?) exceptions
        ON exceptions.gtfs_active_date <= days.service_date AND exceptions.gtfs_end_date >= days.service_date
        AND exceptions.date = days.service_date AND exceptions.exception_type = 1
    ), removed AS (
      SELECT DISTINCT days.service_date, exceptions.service_id
      FROM service_days days
      JOIN read_parquet(?) exceptions
        ON exceptions.gtfs_active_date <= days.service_date AND exceptions.gtfs_end_date >= days.service_date
        AND exceptions.date = days.service_date AND exceptions.exception_type = 2
    )
    (SELECT * FROM regular UNION SELECT * FROM added)
    EXCEPT SELECT * FROM removed
    """,
    [calendar, exceptions, exceptions],
  )
  connection.execute(
    """
    CREATE TEMP TABLE planned_departures AS
    WITH calls AS (
      SELECT *,
        split_part(departure_time, ':', 1)::INT * 3600
          + split_part(departure_time, ':', 2)::INT * 60
          + split_part(departure_time, ':', 3)::INT AS departure
      FROM read_parquet(?)
    )
    SELECT DISTINCT days.service_date, coalesce(stops.parent_station, stops.stop_id) AS parent_station,
      mapping.trunk_route_id, trips.direction_id,
      days.start_timestamp + calls.departure AS departure_timestamp
    FROM service_days days
    JOIN active_services services USING (service_date)
    JOIN read_parquet(?) trips
      ON trips.service_id = services.service_id
      AND trips.gtfs_active_date <= days.service_date AND trips.gtfs_end_date >= days.service_date
    JOIN route_mapping mapping ON mapping.route_id = trips.route_id
    JOIN calls
      ON calls.trip_id = trips.trip_id
      AND calls.gtfs_active_date <= days.service_date AND calls.gtfs_end_date >= days.service_date
    JOIN read_parquet(?) stops
      ON stops.stop_id = calls.stop_id
      AND stops.gtfs_active_date <= days.service_date AND stops.gtfs_end_date >= days.service_date
    """,
    [calls, trips, stops],
  )
  connection.execute(
    """
    CREATE TEMP TABLE plan_intervals AS
    WITH ordered AS (
      SELECT *, lag(departure_timestamp) OVER lane AS previous_timestamp
      FROM planned_departures
      WINDOW lane AS (
        PARTITION BY service_date, parent_station, trunk_route_id, direction_id
        ORDER BY departure_timestamp
      )
    )
    SELECT *, departure_timestamp - previous_timestamp AS headway_seconds
    FROM ordered
    WHERE previous_timestamp IS NOT NULL AND departure_timestamp > previous_timestamp
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE task AS
    SELECT targets.*, plan.headway_seconds AS plan_seconds
    FROM task_targets targets
    ASOF LEFT JOIN plan_intervals plan
      ON targets.service_date = plan.service_date
      AND targets.parent_station = plan.parent_station
      AND targets.trunk_route_id = plan.trunk_route_id
      AND targets.direction_id = plan.direction_id
      AND targets.cutoff_timestamp <= plan.departure_timestamp
    """
  )


def _protocol(connection: Any, source_dir: Path, population_audit: Path, population: dict) -> dict:
  counts = _rows(
    connection,
    """
    SELECT split, count(*) AS targets, count(DISTINCT service_date) AS dates,
      count(DISTINCT route_id) AS routes,
      count(*) FILTER (WHERE schedule_resolved) AS schedule_resolved,
      count(*) FILTER (WHERE NOT schedule_resolved) AS schedule_unresolved,
      count(*) FILTER (WHERE route_aliases > 1) AS multiple_route_aliases,
      count(*) FILTER (WHERE ambiguous_run_source) AS ambiguous_run_sources,
      count(*) FILTER (WHERE ambiguous_run_target) AS ambiguous_run_targets,
      min(elapsed_seconds) AS minimum_seconds,
      quantile_disc(elapsed_seconds, 0.5) AS median_seconds,
      quantile_disc(elapsed_seconds, 0.95) AS p95_seconds,
      max(elapsed_seconds) AS maximum_seconds
    FROM task
    GROUP BY split
    ORDER BY CASE split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END
    """,
  )
  route_counts = _rows(
    connection,
    """
    SELECT split, route_id, count(*) AS targets
    FROM task
    GROUP BY split, route_id
    ORDER BY CASE split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END, route_id
    """,
  )
  return {
    "schema": 1,
    "source_manifest_sha256": hashlib.sha256((source_dir / "manifest.json").read_bytes()).hexdigest(),
    "population_audit_sha256": hashlib.sha256(population_audit.read_bytes()).hexdigest(),
    "availability": "retrospective_event_time_only:no_generation_or_ingestion_clock",
    "identity": ["service_date", "vehicle_id", "parent_station", "direction_id", "departure_timestamp"],
    "lane": ["service_date", "parent_station", "trunk_route_id", "direction_id"],
    "cutoff": "previous_departure_timestamp_plus_one_second",
    "target": "strict_next_movement_headway_seconds",
    "splits": {
      "train": ["2026-07-24", "2026-08-10"],
      "validation": ["2026-08-11", "2026-08-15"],
      "test": ["2026-08-16", "2026-08-20"],
    },
    "target_counts": counts,
    "route_counts": route_counts,
    "population_masks": population,
    "target_masks": [
      "ambiguous_public_order",
      "simultaneous_strict_order",
      "mismatched_source_headway",
      "boundary_only_source_headway",
      "ambiguous_run_source",
      "ambiguous_run_target",
      "unresolved_schedule_identity",
    ],
    "mask_policy": {
      "ambiguous_public_order": {"disposition": "excluded_before_event_lowering", "rows": population["ambiguous_order_rows"]},
      "simultaneous_strict_order": {"disposition": "excluded_from_strict_target", "events": population["simultaneous_events"]},
      "mismatched_source_headway": {"disposition": "excluded_from_exact_target", "labels": population["mismatched_headways"]},
      "boundary_only_source_headway": {"disposition": "excluded_from_exact_target", "labels": population["boundary_only_headways"]},
      "ambiguous_run_source": {"disposition": "retained_annotation"},
      "ambiguous_run_target": {"disposition": "retained_annotation"},
      "unresolved_schedule_identity": {"disposition": "retained_annotation"},
    },
    "metrics": ["mae_seconds", "rmse_seconds", "median_absolute_error", "p90_absolute_error", "coverage"],
    "seeds": [0, 1, 2],
    "references": REFERENCES,
  }


def _temporal_candidate(connection: Any, bin_hours: int, minimum_support: int) -> dict:
  _fit_temporal(connection, bin_hours)
  row = connection.execute(
    """
    SELECT avg(abs(coalesce(
      CASE WHEN cell.samples >= ? THEN cell.median_seconds END,
      CASE WHEN lane.samples >= ? THEN lane.median_seconds END,
      CASE WHEN route.samples >= ? THEN route.median_seconds END,
      global.median_seconds
    ) - target.elapsed_seconds)) AS mae_seconds
    FROM task target
    LEFT JOIN temporal_cell cell
      ON cell.parent_station = target.parent_station AND cell.trunk_route_id = target.trunk_route_id
      AND cell.direction_id = target.direction_id AND cell.weekday = target.weekday
      AND cell.hour_bin = floor(target.local_hour / ?)
    LEFT JOIN temporal_lane lane
      ON lane.parent_station = target.parent_station AND lane.trunk_route_id = target.trunk_route_id
      AND lane.direction_id = target.direction_id
    LEFT JOIN temporal_route route ON route.route_id = target.route_id
    CROSS JOIN temporal_global global
    WHERE target.split = 'validation'
    """,
    [minimum_support, minimum_support, minimum_support, bin_hours],
  ).fetchone()
  return {"bin_hours": bin_hours, "minimum_support": minimum_support, "mae_seconds": round(row[0], 6)}


def _fit_temporal(connection: Any, bin_hours: int) -> None:
  for table in ("temporal_cell", "temporal_lane", "temporal_route", "temporal_global"):
    connection.execute(f"DROP TABLE IF EXISTS {table}")
  connection.execute(
    """
    CREATE TEMP TABLE temporal_cell AS
    SELECT parent_station, trunk_route_id, direction_id, weekday,
      floor(local_hour / ?) AS hour_bin,
      median(elapsed_seconds) AS median_seconds, count(*) AS samples
    FROM task WHERE split = 'train'
    GROUP BY parent_station, trunk_route_id, direction_id, weekday, hour_bin
    """,
    [bin_hours],
  )
  connection.execute(
    """
    CREATE TEMP TABLE temporal_lane AS
    SELECT parent_station, trunk_route_id, direction_id,
      median(elapsed_seconds) AS median_seconds, count(*) AS samples
    FROM task WHERE split = 'train'
    GROUP BY parent_station, trunk_route_id, direction_id
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE temporal_route AS
    SELECT route_id, median(elapsed_seconds) AS median_seconds, count(*) AS samples
    FROM task WHERE split = 'train'
    GROUP BY route_id
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE temporal_global AS
    SELECT median(elapsed_seconds) AS median_seconds, count(*) AS samples
    FROM task WHERE split = 'train'
    """
  )


def _evidence(connection: Any, split: str, selected: dict, candidates: list[dict] | None) -> dict:
  bin_hours = selected["bin_hours"]
  support = selected["minimum_support"]
  _fit_temporal(connection, bin_hours)
  connection.execute("DROP TABLE IF EXISTS predictions")
  connection.execute(
    """
    CREATE TEMP TABLE predictions AS
    SELECT target.*,
      coalesce(
        CASE WHEN cell.samples >= ? THEN cell.median_seconds END,
        CASE WHEN lane.samples >= ? THEN lane.median_seconds END,
        CASE WHEN route.samples >= ? THEN route.median_seconds END,
        global.median_seconds
      ) AS temporal_seconds,
      CASE
        WHEN cell.samples >= ? THEN 'cell'
        WHEN lane.samples >= ? THEN 'lane'
        WHEN route.samples >= ? THEN 'route'
        ELSE 'global'
      END AS temporal_fallback
    FROM task target
    LEFT JOIN temporal_cell cell
      ON cell.parent_station = target.parent_station AND cell.trunk_route_id = target.trunk_route_id
      AND cell.direction_id = target.direction_id AND cell.weekday = target.weekday
      AND cell.hour_bin = floor(target.local_hour / ?)
    LEFT JOIN temporal_lane lane
      ON lane.parent_station = target.parent_station AND lane.trunk_route_id = target.trunk_route_id
      AND lane.direction_id = target.direction_id
    LEFT JOIN temporal_route route ON route.route_id = target.route_id
    CROSS JOIN temporal_global global
    WHERE target.split = ?
    """,
    [support, support, support, support, support, support, bin_hours, split],
  )
  metrics = [_metrics(connection, baseline) for baseline in ("persistence", "temporal", "plan")]
  routes = [row | {"mae_seconds": round(row["mae_seconds"], 6)} for row in _rows(
    connection,
    """
    WITH values AS (
      SELECT route_id, elapsed_seconds, persistence_seconds AS persistence,
        temporal_seconds AS temporal, plan_seconds AS plan
      FROM predictions
    ), long AS (
      SELECT route_id, elapsed_seconds, unnest(['persistence', 'temporal', 'plan']) AS baseline,
        unnest([persistence, temporal, plan]) AS prediction
      FROM values
    )
    SELECT baseline, route_id, count(prediction) AS predictions,
      count(*) AS targets, avg(abs(prediction - elapsed_seconds)) AS mae_seconds
    FROM long
    GROUP BY baseline, route_id
    ORDER BY baseline, route_id
    """,
  )]
  fallback = _rows(
    connection,
    """
    SELECT temporal_fallback AS level, count(*) AS predictions
    FROM predictions GROUP BY temporal_fallback ORDER BY temporal_fallback
    """,
  )
  schedule = _rows(
    connection,
    """
    SELECT schedule_resolved, count(*) AS targets, count(plan_seconds) AS predictions
    FROM predictions GROUP BY schedule_resolved ORDER BY schedule_resolved DESC
    """,
  )
  result = {
    "schema": 1,
    "split": split,
    "selected_temporal": {"bin_hours": bin_hours, "minimum_support": support},
    "metrics": metrics,
    "route_metrics": routes,
    "temporal_fallback": fallback,
    "plan_coverage_by_schedule_identity": schedule,
    "decision": "freeze:stage_4_input" if split == "test" else "freeze:open_test_once",
  }
  if candidates is not None:
    result["temporal_candidates"] = candidates
  return result


def _metrics(connection: Any, baseline: str) -> dict:
  column = f"{baseline}_seconds"
  row = connection.execute(
    f"""
    WITH errors AS (
      SELECT abs({column} - elapsed_seconds) AS absolute_error,
        ({column} - elapsed_seconds) * ({column} - elapsed_seconds) AS squared_error
      FROM predictions WHERE {column} IS NOT NULL
    )
    SELECT (SELECT count({column}) FROM predictions) AS predictions,
      (SELECT count(*) FROM predictions) AS targets,
      avg(absolute_error) AS mae_seconds,
      sqrt(avg(squared_error)) AS rmse_seconds,
      quantile_disc(absolute_error, 0.5) AS median_absolute_error,
      quantile_disc(absolute_error, 0.9) AS p90_absolute_error
    FROM errors
    """
  ).fetchone()
  predictions, targets, mae, rmse, median_error, p90_error = row
  route_mae = connection.execute(
    f"""
    SELECT avg(mae) FROM (
      SELECT route_id, avg(abs({column} - elapsed_seconds)) AS mae
      FROM predictions WHERE {column} IS NOT NULL GROUP BY route_id
    )
    """
  ).fetchone()[0]
  return {
    "baseline": baseline,
    "predictions": predictions,
    "targets": targets,
    "coverage": round(predictions / targets, 6),
    "mae_seconds": round(mae, 6),
    "rmse_seconds": round(rmse, 6),
    "median_absolute_error": median_error,
    "p90_absolute_error": p90_error,
    "macro_route_mae_seconds": round(route_mae, 6),
  }


def _rows(connection: Any, query: str) -> list[dict[str, Any]]:
  cursor = connection.execute(query)
  columns = tuple(column[0] for column in cursor.description)
  return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _source(directory: Path, plan: Plan, name: str) -> Path:
  source = next((source for source in plan.sources if source.name == name), None)
  if source is None:
    raise TaskError(f"manifest: missing {name}")
  return directory / source.filename


def _write(path: Path, value: object) -> None:
  encoded = _encode(value)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(path)


def _encode(value: object) -> str:
  return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def _digest(value: object) -> str:
  return hashlib.sha256(_encode(value).encode()).hexdigest()


def _read(path: Path) -> dict:
  try:
    value = json.loads(path.read_bytes())
  except (FileNotFoundError, json.JSONDecodeError) as error:
    raise TaskError(f"{path.name}: missing or invalid frozen artifact") from error
  if not isinstance(value, dict):
    raise TaskError(f"{path.name}: frozen artifact is not an object")
  return value


def _freeze(path: Path, value: object) -> None:
  if path.exists():
    if path.read_text() != _encode(value):
      raise TaskError(f"{path.name}: frozen artifact drift")
    return
  _write(path, value)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--population-audit", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--test", action="store_true")
  arguments = parser.parse_args()
  arguments.output_dir.mkdir(parents=True, exist_ok=True)
  protocol_path = arguments.output_dir / "protocol.json"
  validation_path = arguments.output_dir / "validation.json"
  test_path = arguments.output_dir / "test.json"
  if arguments.test and test_path.exists():
    raise TaskError("test.json: test split was already opened")
  frozen_protocol = _read(protocol_path) if arguments.test else None
  frozen_validation = _read(validation_path) if arguments.test else None
  protocol, validation, test = build(
    arguments.source_dir,
    arguments.population_audit,
    arguments.test,
    frozen_protocol,
    frozen_validation,
  )
  _freeze(protocol_path, protocol)
  _freeze(validation_path, validation)
  if test is not None:
    _write(test_path, test)
  print(json.dumps({"targets": sum(row["targets"] for row in protocol["target_counts"]), "test": test is not None}))


if __name__ == "__main__":
  main()
