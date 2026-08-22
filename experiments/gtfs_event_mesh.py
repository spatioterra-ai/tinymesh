"""Prove a reversible causal carrier for the retained MBTA departure events."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from experiments.gtfs_replay import FIXTURE, ReplayRow, RetainedReplay, ScheduleCall, TripInstance, load


RelationKind = Literal["headway", "run"]


class EventMeshError(ValueError):
  """A departure event or causal relation violates its contract."""


@dataclass(frozen=True, order=True)
class EventKey:
  service_date: int
  vehicle_id: str
  parent_station: str
  direction_id: int
  departure_timestamp: int


@dataclass(frozen=True)
class SourceAlias:
  instance: TripInstance
  stop_id: str
  stop_sequence: int
  following_stop_id: str
  following_stop_sequence: int
  source_headway_seconds: int | None
  scheduled_departure_time: int | None

  @property
  def source_identity(self) -> tuple[TripInstance, int]:
    return self.instance, self.stop_sequence


@dataclass(frozen=True)
class DepartureEvent:
  key: EventKey
  trunk_route_id: str
  aliases: tuple[SourceAlias, ...]


@dataclass(frozen=True, order=True)
class Relation:
  kind: RelationKind
  source: EventKey
  target: EventKey
  elapsed_seconds: int


@dataclass(frozen=True)
class EventMesh:
  events: tuple[DepartureEvent, ...]
  relations: tuple[Relation, ...]

  def __post_init__(self) -> None:
    keys = tuple(event.key for event in self.events)
    if len(keys) != len(set(keys)):
      raise EventMeshError("event: duplicate physical identity")
    if any(not event.aliases for event in self.events):
      raise EventMeshError("event: physical departure has no source alias")

    known = set(keys)
    identities: set[tuple[RelationKind, EventKey, EventKey]] = set()
    endpoints: dict[tuple[EventKey, EventKey], RelationKind] = {}
    for relation in self.relations:
      identity = relation.kind, relation.source, relation.target
      if relation.source not in known or relation.target not in known:
        raise EventMeshError(f"relation: unresolved endpoint {identity!r}")
      if relation.source == relation.target:
        raise EventMeshError(f"relation: self edge {identity!r}")
      if relation.source.departure_timestamp >= relation.target.departure_timestamp:
        raise EventMeshError(f"relation: time did not advance {identity!r}")
      if relation.elapsed_seconds != relation.target.departure_timestamp - relation.source.departure_timestamp:
        raise EventMeshError(f"relation: contradictory elapsed time {identity!r}")
      if identity in identities:
        raise EventMeshError(f"relation: duplicate {identity!r}")
      other = endpoints.get((relation.source, relation.target))
      if other is not None and other != relation.kind:
        raise EventMeshError(f"relation: contradictory kinds {(other, relation.kind, relation.source, relation.target)!r}")
      identities.add(identity)
      endpoints[(relation.source, relation.target)] = relation.kind

  def prefix(self, cutoff: int) -> EventMesh:
    """Return exactly the event-induced subgraph known before ``cutoff``."""
    events = tuple(event for event in self.events if event.key.departure_timestamp < cutoff)
    retained = {event.key for event in events}
    relations = tuple(
      relation for relation in self.relations if relation.source in retained and relation.target in retained
    )
    return EventMesh(events, relations)


@dataclass(frozen=True)
class MeshAudit:
  source_rows: int
  represented_source_rows: int
  physical_departures: int
  source_aliases: int
  duplicate_aliases: int
  conflicting_aliases: int
  run_relations: int
  headway_relations: int
  source_headways: int
  exact_headways: int
  derived_only_headways: int
  boundary_only_headways: int
  station_directions: int
  max_in_degree: int
  max_out_degree: int
  event_records: int
  relation_records: int
  sparse_records: int
  dense_event_pair_cells: int


@dataclass(frozen=True)
class CutAudit:
  cutoff: int
  retained_events: int
  retained_relations: int
  excluded_events: int
  crossing_relations: int


@dataclass(frozen=True)
class ClockAudit:
  seconds: int
  cells: int
  occupied_cells: int
  empty_cells: int
  target_events: int
  colliding_cells: int
  merged_events: int
  max_events_per_cell: int
  aggregation_required: bool
  identity_slots_without_sidecar: int


@dataclass(frozen=True)
class Observation:
  mesh: MeshAudit
  midpoint_prefix: CutAudit
  clocks: tuple[ClockAudit, ...]
  carrier_decision: str
  stage_2_consequence: str


def lower(source: RetainedReplay) -> EventMesh:
  """Lower validated replay facts into physical departures and typed edges."""
  calls = {(call.trip_id, call.stop_sequence): call for call in source.calls}
  following = _following(source.calls)
  _validate_rows(source.rows, calls)

  trips: dict[tuple[TripInstance, str], list[ReplayRow]] = defaultdict(list)
  for row in source.rows:
    trips[(row.instance, row.vehicle_id)].append(row)

  candidates: list[tuple[EventKey, str, SourceAlias]] = []
  for rows in trips.values():
    ordered = sorted(rows, key=lambda row: row.stop_sequence)
    for current, next_row in zip(ordered, ordered[1:]):
      expected = following.get((current.instance.trip_id, current.stop_sequence))
      if expected != next_row.stop_sequence or next_row.move_timestamp is None:
        continue
      key = EventKey(
        current.instance.service_date,
        current.vehicle_id,
        current.parent_station,
        current.direction_id,
        next_row.move_timestamp,
      )
      alias = SourceAlias(
        current.instance,
        current.stop_id,
        current.stop_sequence,
        next_row.stop_id,
        next_row.stop_sequence,
        current.headway_trunk_seconds,
        current.scheduled_departure_time,
      )
      candidates.append((key, current.trunk_route_id, alias))

  physical: dict[EventKey, list[tuple[str, SourceAlias]]] = defaultdict(list)
  for key, trunk_route_id, alias in candidates:
    physical[key].append((trunk_route_id, alias))

  events = []
  alias_event: dict[tuple[TripInstance, str, int], EventKey] = {}
  for key, grouped in physical.items():
    routes = {route for route, _ in grouped}
    labels = {alias.source_headway_seconds for _, alias in grouped if alias.source_headway_seconds is not None}
    if len(routes) != 1 or len(labels) > 1:
      identities = tuple(alias.source_identity for _, alias in grouped)
      raise EventMeshError(f"alias: conflicting physical departure {key!r} sources={identities!r}")
    aliases = tuple(sorted((alias for _, alias in grouped), key=_alias_order))
    event = DepartureEvent(key, next(iter(routes)), aliases)
    events.append(event)
    for alias in aliases:
      alias_event[(alias.instance, key.vehicle_id, alias.stop_sequence)] = key

  events.sort(key=lambda event: (event.key.departure_timestamp, event.key))
  relations = _run_relations(trips, alias_event, following)
  relations.extend(_headway_relations(events))
  relations.sort(key=lambda relation: (relation.target.departure_timestamp, relation.kind, relation.source, relation.target))
  mesh = EventMesh(tuple(events), tuple(relations))
  _validate_headways(mesh)
  return mesh


def observe(path: str | Path = FIXTURE) -> Observation:
  source = load(path)
  mesh = lower(source)
  events = {event.key: event for event in mesh.events}
  headways = tuple(relation for relation in mesh.relations if relation.kind == "headway")
  exact = tuple(relation for relation in headways if _source_labels(events, relation.target))
  represented = {alias.source_identity for event in mesh.events for alias in event.aliases}
  source_labels = {
    (row.instance, row.stop_sequence)
    for row in source.rows
    if row.headway_trunk_seconds is not None
  }
  exact_labels = {
    alias.source_identity
    for relation in exact
    for alias in events[relation.target].aliases
    if alias.source_headway_seconds is not None
  }
  lanes = {
    (event.key.parent_station, event.trunk_route_id, event.key.direction_id)
    for relation in exact
    for event in (events[relation.target],)
  }
  in_degree: dict[EventKey, int] = defaultdict(int)
  out_degree: dict[EventKey, int] = defaultdict(int)
  for relation in mesh.relations:
    in_degree[relation.target] += 1
    out_degree[relation.source] += 1
  audit = MeshAudit(
    source_rows=len(source.rows),
    represented_source_rows=len(represented),
    physical_departures=len(mesh.events),
    source_aliases=sum(len(event.aliases) for event in mesh.events),
    duplicate_aliases=sum(len(event.aliases) - 1 for event in mesh.events),
    conflicting_aliases=0,
    run_relations=sum(relation.kind == "run" for relation in mesh.relations),
    headway_relations=len(headways),
    source_headways=len(source_labels),
    exact_headways=len(exact_labels),
    derived_only_headways=sum(not _source_labels(events, relation.target) for relation in headways),
    boundary_only_headways=len(source_labels - exact_labels),
    station_directions=len(lanes),
    max_in_degree=max(in_degree.values(), default=0),
    max_out_degree=max(out_degree.values(), default=0),
    event_records=len(mesh.events),
    relation_records=len(mesh.relations),
    sparse_records=len(mesh.events) + len(mesh.relations),
    dense_event_pair_cells=len(mesh.events) ** 2,
  )
  clocks = tuple(_clock(events, exact, source.interval_utc, seconds) for seconds in (30, 60, 300))
  midpoint = sum(source.interval_utc) // 2
  return Observation(
    mesh=audit,
    midpoint_prefix=_cut_audit(mesh, midpoint),
    clocks=clocks,
    carrier_decision="retain:event_mesh",
    stage_2_consequence="retain_event_facts_and_boundaries;derive_clocks_only_for_matched_controls",
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _validate_rows(rows: tuple[ReplayRow, ...], calls: dict[tuple[str, int], ScheduleCall]) -> None:
  for row in rows:
    identity = row.instance, row.stop_sequence
    call = calls.get((row.instance.trip_id, row.stop_sequence))
    if call is None:
      raise EventMeshError(f"schedule: unresolved source {identity!r}")
    if (call.stop_id, call.arrival_time, call.departure_time) != (
      row.stop_id,
      row.scheduled_arrival_time,
      row.scheduled_departure_time,
    ):
      raise EventMeshError(f"schedule: contradictory source {identity!r}")


def _following(calls: tuple[ScheduleCall, ...]) -> dict[tuple[str, int], int]:
  grouped: dict[str, list[ScheduleCall]] = defaultdict(list)
  for call in calls:
    grouped[call.trip_id].append(call)
  return {
    (trip_id, current.stop_sequence): following.stop_sequence
    for trip_id, trip_calls in grouped.items()
    for current, following in zip(sorted(trip_calls), sorted(trip_calls)[1:])
  }


def _run_relations(
  trips: dict[tuple[TripInstance, str], list[ReplayRow]],
  alias_event: dict[tuple[TripInstance, str, int], EventKey],
  following: dict[tuple[str, int], int],
) -> list[Relation]:
  edges: set[tuple[EventKey, EventKey]] = set()
  for (instance, vehicle_id), rows in trips.items():
    keys = [
      (row.stop_sequence, alias_event[(instance, vehicle_id, row.stop_sequence)])
      for row in sorted(rows, key=lambda row: row.stop_sequence)
      if (instance, vehicle_id, row.stop_sequence) in alias_event
    ]
    for (source_sequence, source), (target_sequence, target) in zip(keys, keys[1:]):
      if following.get((instance.trip_id, source_sequence)) == target_sequence:
        edges.add((source, target))

  successors: dict[EventKey, set[EventKey]] = defaultdict(set)
  predecessors: dict[EventKey, set[EventKey]] = defaultdict(set)
  for source, target in edges:
    successors[source].add(target)
    predecessors[target].add(source)
  conflicts = [key for key, values in (*successors.items(), *predecessors.items()) if len(values) > 1]
  if conflicts:
    raise EventMeshError(f"alias: contradictory run continuation {min(conflicts)!r}")
  return [Relation("run", source, target, target.departure_timestamp - source.departure_timestamp) for source, target in edges]


def _headway_relations(events: list[DepartureEvent]) -> list[Relation]:
  lanes: dict[tuple[str, str, int], list[DepartureEvent]] = defaultdict(list)
  for event in events:
    lanes[(event.key.parent_station, event.trunk_route_id, event.key.direction_id)].append(event)
  relations = []
  for lane in lanes.values():
    ordered = sorted(lane, key=lambda event: (event.key.departure_timestamp, event.key))
    relations.extend(
      Relation("headway", previous.key, current.key, current.key.departure_timestamp - previous.key.departure_timestamp)
      for previous, current in zip(ordered, ordered[1:])
    )
  return relations


def _validate_headways(mesh: EventMesh) -> None:
  events = {event.key: event for event in mesh.events}
  for relation in mesh.relations:
    if relation.kind != "headway":
      continue
    labels = _source_labels(events, relation.target)
    if labels and labels != {relation.elapsed_seconds}:
      event = events[relation.target]
      sources = tuple(alias.source_identity for alias in event.aliases)
      raise EventMeshError(f"headway: source mismatch target={relation.target!r} sources={sources!r}")


def _source_labels(events: dict[EventKey, DepartureEvent], key: EventKey) -> set[int]:
  return {
    alias.source_headway_seconds
    for alias in events[key].aliases
    if alias.source_headway_seconds is not None
  }


def _cut_audit(mesh: EventMesh, cutoff: int) -> CutAudit:
  prefix = mesh.prefix(cutoff)
  retained = {event.key for event in prefix.events}
  return CutAudit(
    cutoff=cutoff,
    retained_events=len(prefix.events),
    retained_relations=len(prefix.relations),
    excluded_events=len(mesh.events) - len(prefix.events),
    crossing_relations=sum(
      (relation.source in retained) != (relation.target in retained)
      for relation in mesh.relations
    ),
  )


def _clock(
  events: dict[EventKey, DepartureEvent],
  targets: tuple[Relation, ...],
  interval: tuple[int, int],
  seconds: int,
) -> ClockAudit:
  start, end = interval
  lanes = {
    (event.key.parent_station, event.trunk_route_id, event.key.direction_id)
    for relation in targets
    for event in (events[relation.target],)
  }
  occupied: dict[tuple[str, str, int, int], int] = defaultdict(int)
  for relation in targets:
    event = events[relation.target]
    lane = event.key.parent_station, event.trunk_route_id, event.key.direction_id
    occupied[(*lane, (event.key.departure_timestamp - start) // seconds)] += 1
  cells = len(lanes) * ((end - start + seconds - 1) // seconds)
  return ClockAudit(
    seconds=seconds,
    cells=cells,
    occupied_cells=len(occupied),
    empty_cells=cells - len(occupied),
    target_events=len(targets),
    colliding_cells=sum(count > 1 for count in occupied.values()),
    merged_events=sum(count - 1 for count in occupied.values()),
    max_events_per_cell=max(occupied.values(), default=0),
    aggregation_required=any(count > 1 for count in occupied.values()),
    identity_slots_without_sidecar=len(occupied),
  )


def _alias_order(alias: SourceAlias) -> tuple[TripInstance, int, str]:
  return alias.instance, alias.stop_sequence, alias.stop_id


if __name__ == "__main__":
  main()
