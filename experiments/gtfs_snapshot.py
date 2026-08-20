"""Lower validated GTFS host facts into one sparse TinyMesh snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Device, Tensor, dtypes

from tinymesh import Graph

from experiments.gtfs_realtime import FIXTURE, Snapshot, normalize
from experiments.gtfs_schedule import RouteSegment, Schedule, SegmentOccurrence, load as load_schedule
from experiments.gtfs_transition import Transition, TransitionPolicy, VehicleKey, evaluate, fingerprint


class ProjectionError(ValueError):
  """An invalid sparse-snapshot projection input."""


@dataclass(frozen=True)
class EdgeRow:
  position: int
  route_id: str
  from_stop_id: str
  to_stop_id: str
  occurrences: tuple[SegmentOccurrence, ...]


@dataclass(frozen=True)
class Projection:
  graph_version: str
  graph: Graph
  node_ids: tuple[str, ...]
  edges: tuple[EdgeRow, ...]
  delay: Tensor
  observed: Tensor
  vehicle_count: Tensor


@dataclass(frozen=True)
class Observation:
  device: str
  graph_version: str
  nodes: int
  edges: int
  delay_shape: tuple[int, ...]
  observed_shape: tuple[int, ...]
  vehicle_count_shape: tuple[int, ...]
  observed_delays: int
  vehicles: int


def lower(
  schedule: Schedule,
  snapshot: Snapshot | None = None,
  transition: Transition | None = None,
  *,
  device: str = Device.DEFAULT,
) -> Projection:
  """Derive sparse topology and one aligned observation without changing host truth."""
  if (snapshot is None) != (transition is None):
    raise ProjectionError("snapshot and transition must be supplied together")
  node_ids = tuple(sorted(stop.stop_id for stop in schedule.stops))
  rows = {stop_id: index for index, stop_id in enumerate(node_ids)}
  segments = tuple(sorted(schedule.segments, key=lambda item: (item.route_id, item.from_stop_id, item.to_stop_id)))
  edges = tuple(
    EdgeRow(index, segment.route_id, segment.from_stop_id, segment.to_stop_id, segment.occurrences)
    for index, segment in enumerate(segments)
  )
  graph = Graph(
    len(node_ids),
    [rows[segment.from_stop_id] for segment in segments],
    [rows[segment.to_stop_id] for segment in segments],
  )
  delay = [0.0] * len(node_ids)
  observed = [False] * len(node_ids)
  vehicles = [0] * len(node_ids)
  if snapshot is not None and transition is not None:
    _attach(schedule, snapshot, transition, rows, delay, observed, vehicles)
  return Projection(
    _version(schedule, segments),
    graph,
    node_ids,
    edges,
    Tensor(delay, dtype=dtypes.float32, device=device).reshape(len(node_ids), 1).realize(),
    Tensor(observed, dtype=dtypes.bool, device=device).reshape(len(node_ids), 1).realize(),
    Tensor(vehicles, dtype=dtypes.int32, device=device).reshape(len(node_ids), 1).realize(),
  )


def observe(path: str | Path = FIXTURE, *, device: str = Device.DEFAULT) -> Observation:
  schedule = load_schedule()
  snapshot = normalize(json.loads(Path(path).read_text()), schedule, schedule.source_id)
  transition = evaluate(snapshot, snapshot, snapshot.generated_at, TransitionPolicy(30, 90, 90))
  projection = lower(schedule, snapshot, transition, device=device)
  return Observation(
    projection.delay.device,
    projection.graph_version,
    projection.graph.nodes,
    projection.graph.edges,
    projection.delay.shape,
    projection.observed.shape,
    projection.vehicle_count.shape,
    int(projection.observed.sum().item()),
    int(projection.vehicle_count.sum().item()),
  )


def main() -> None:
  print(json.dumps(asdict(observe()), indent=2))


def _attach(
  schedule: Schedule,
  snapshot: Snapshot,
  transition: Transition,
  rows: dict[str, int],
  delay: list[float],
  observed: list[bool],
  vehicles: list[int],
) -> None:
  if snapshot.source_id != schedule.source_id:
    raise ProjectionError(f"snapshot source {snapshot.source_id!r} does not match schedule {schedule.source_id!r}")
  if (snapshot.schedule_revision, snapshot.schedule_sha256) != (schedule.revision, schedule.sha256):
    raise ProjectionError("snapshot does not resolve against the supplied schedule manifest")
  if transition.current_fingerprint != fingerprint(snapshot):
    raise ProjectionError("transition does not describe the supplied snapshot")
  trip_by_instance = {trip.instance: trip for trip in snapshot.trips}
  vehicle_by_key = {VehicleKey(vehicle.instance, vehicle.vehicle_id): vehicle for vehicle in snapshot.vehicles}
  eligible_trips = set(transition.eligible_trips)
  eligible_vehicles = set(transition.eligible_vehicles)
  if not eligible_trips <= set(trip_by_instance) or not eligible_vehicles <= set(vehicle_by_key):
    raise ProjectionError("transition eligibility references absent snapshot facts")
  if len(eligible_vehicles) > 1:
    raise ProjectionError("Stage 1 supports one eligible vehicle")
  for key in eligible_vehicles:
    vehicle = vehicle_by_key[key]
    row = rows[vehicle.current_stop_id]
    vehicles[row] = 1
    trip = trip_by_instance.get(vehicle.instance)
    if vehicle.instance in eligible_trips and trip is not None and trip.relationship == "SCHEDULED" and trip.delay_seconds is not None:
      delay[row] = float(trip.delay_seconds)
      observed[row] = True


def _version(schedule: Schedule, segments: tuple[RouteSegment, ...]) -> str:
  manifest = {
    "source_id": schedule.source_id,
    "revision": schedule.revision,
    "sha256": schedule.sha256,
    "segments": [asdict(segment) for segment in segments],
  }
  return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
  main()
