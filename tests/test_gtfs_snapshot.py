import json
import unittest
from dataclasses import replace

from tinygrad import dtypes

from experiments.gtfs_realtime import FIXTURE, normalize
from experiments.gtfs_schedule import RouteSegment, SegmentOccurrence, parse
from experiments.gtfs_snapshot import ProjectionError, lower
from experiments.gtfs_transition import TransitionPolicy, VehicleKey, evaluate, fingerprint
from tests.test_gtfs_schedule import archive


SOURCE_ID = "google-transit/sample-feed-1"
POLICY = TransitionPolicy(30, 90, 90)


class GtfsSnapshotTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.schedule = parse(archive(), source_id=SOURCE_ID)
    cls.snapshot = normalize(json.loads(FIXTURE.read_text()), cls.schedule, SOURCE_ID)
    cls.transition = evaluate(cls.snapshot, cls.snapshot, cls.snapshot.generated_at, POLICY)

  def test_matches_independent_host_topology_and_fields(self) -> None:
    projection = lower(self.schedule, self.snapshot, self.transition, device="CPU")

    node_ids = tuple(sorted(stop.stop_id for stop in self.schedule.stops))
    expected_segments = tuple(sorted(self.schedule.segments, key=lambda item: (item.route_id, item.from_stop_id, item.to_stop_id)))
    expected_delay = {stop_id: 0.0 for stop_id in node_ids}
    expected_observed = {stop_id: False for stop_id in node_ids}
    expected_vehicles = {stop_id: 0 for stop_id in node_ids}
    expected_delay["NANAA"] = 120.0
    expected_observed["NANAA"] = True
    expected_vehicles["NANAA"] = 1

    self.assertEqual(projection.graph.nodes, len(node_ids))
    self.assertEqual(projection.graph.edges, len(expected_segments))
    self.assertEqual(projection.node_ids, node_ids)
    self.assertEqual(
      [projection.node_ids[row] for row in range(projection.graph.nodes)],
      list(node_ids),
    )
    for position, (edge, segment, source, target) in enumerate(
      zip(projection.edges, expected_segments, projection.graph.source, projection.graph.target)
    ):
      self.assertEqual(edge.position, position)
      self.assertEqual((edge.route_id, edge.from_stop_id, edge.to_stop_id), (segment.route_id, segment.from_stop_id, segment.to_stop_id))
      self.assertEqual(edge.occurrences, segment.occurrences)
      self.assertEqual((projection.node_ids[source], projection.node_ids[target]), (segment.from_stop_id, segment.to_stop_id))
    self.assertEqual(dict(zip(node_ids, (row[0] for row in projection.delay.tolist()))), expected_delay)
    self.assertEqual(dict(zip(node_ids, (row[0] for row in projection.observed.tolist()))), expected_observed)
    self.assertEqual(dict(zip(node_ids, (row[0] for row in projection.vehicle_count.tolist()))), expected_vehicles)

  def test_uses_sparse_linear_shapes_and_explicit_dtypes(self) -> None:
    projection = lower(self.schedule, self.snapshot, self.transition, device="CPU")

    self.assertEqual(projection.delay.shape, (projection.graph.nodes, 1))
    self.assertEqual(projection.observed.shape, (projection.graph.nodes, 1))
    self.assertEqual(projection.vehicle_count.shape, (projection.graph.nodes, 1))
    self.assertEqual((projection.delay.dtype, projection.observed.dtype, projection.vehicle_count.dtype), (dtypes.float32, dtypes.bool, dtypes.int32))
    self.assertEqual(len(projection.graph.source), projection.graph.edges)
    self.assertEqual(len(projection.edges), projection.graph.edges)

  def test_source_reordering_is_canonical(self) -> None:
    permuted = replace(self.schedule, stops=tuple(reversed(self.schedule.stops)), segments=tuple(reversed(self.schedule.segments)))

    original = lower(self.schedule, self.snapshot, self.transition, device="CPU")
    reordered = lower(permuted, self.snapshot, self.transition, device="CPU")

    self.assertEqual(original.graph_version, reordered.graph_version)
    self.assertEqual(original.graph, reordered.graph)
    self.assertEqual(original.node_ids, reordered.node_ids)
    self.assertEqual(original.edges, reordered.edges)
    self.assertEqual(original.delay.tolist(), reordered.delay.tolist())
    self.assertEqual(original.observed.tolist(), reordered.observed.tolist())
    self.assertEqual(original.vehicle_count.tolist(), reordered.vehicle_count.tolist())

  def test_stop_relabeling_relabels_topology_and_fields(self) -> None:
    original = lower(self.schedule, self.snapshot, self.transition, device="CPU")
    labels = list(original.node_ids)
    mapping = {stop_id: labels[(index + 3) % len(labels)] for index, stop_id in enumerate(labels)}
    stops = tuple(replace(stop, stop_id=mapping[stop.stop_id]) for stop in self.schedule.stops)
    segments = tuple(
      replace(segment, from_stop_id=mapping[segment.from_stop_id], to_stop_id=mapping[segment.to_stop_id])
      for segment in self.schedule.segments
    )
    schedule = replace(self.schedule, stops=stops, segments=segments)
    trips = tuple(
      replace(
        trip,
        predictions=tuple(replace(prediction, stop_id=mapping[prediction.stop_id]) for prediction in trip.predictions),
      )
      for trip in self.snapshot.trips
    )
    vehicles = tuple(replace(vehicle, current_stop_id=mapping[vehicle.current_stop_id]) for vehicle in self.snapshot.vehicles)
    snapshot = replace(self.snapshot, trips=trips, vehicles=vehicles)
    transition = replace(self.transition, current_fingerprint=fingerprint(snapshot))

    relabeled = lower(schedule, snapshot, transition, device="CPU")

    expected_edges = {(edge.route_id, mapping[edge.from_stop_id], mapping[edge.to_stop_id]) for edge in original.edges}
    self.assertEqual({(edge.route_id, edge.from_stop_id, edge.to_stop_id) for edge in relabeled.edges}, expected_edges)
    delay = dict(zip(relabeled.node_ids, (row[0] for row in relabeled.delay.tolist())))
    observed = dict(zip(relabeled.node_ids, (row[0] for row in relabeled.observed.tolist())))
    self.assertEqual(delay[mapping["NANAA"]], 120.0)
    self.assertTrue(observed[mapping["NANAA"]])
    self.assertNotEqual(original.graph_version, relabeled.graph_version)

  def test_parallel_routes_and_repeated_occurrences_remain_recoverable(self) -> None:
    segment = self.schedule.segments[0]
    occurrence = SegmentOccurrence("parallel-trip", 7, 8)
    repeated = replace(segment, occurrences=(*segment.occurrences, occurrence))
    parallel = RouteSegment("PARALLEL", segment.from_stop_id, segment.to_stop_id, (occurrence,))
    schedule = replace(self.schedule, segments=(*self.schedule.segments[1:], repeated, parallel))

    projection = lower(schedule, device="CPU")
    matching = [edge for edge in projection.edges if (edge.from_stop_id, edge.to_stop_id) == (segment.from_stop_id, segment.to_stop_id)]

    self.assertEqual({edge.route_id for edge in matching}, {segment.route_id, "PARALLEL"})
    self.assertIn(occurrence, next(edge for edge in matching if edge.route_id == segment.route_id).occurrences)
    self.assertEqual(next(edge for edge in matching if edge.route_id == "PARALLEL").occurrences, (occurrence,))
    positions = [edge.position for edge in matching]
    self.assertEqual(
      {(projection.graph.source[position], projection.graph.target[position]) for position in positions},
      {(projection.graph.source[positions[0]], projection.graph.target[positions[0]])},
    )

  def test_changed_call_occurrence_changes_graph_version_without_changing_topology(self) -> None:
    original = lower(self.schedule, device="CPU")
    segment = self.schedule.segments[0]
    occurrence = segment.occurrences[0]
    changed = replace(
      segment,
      occurrences=(replace(occurrence, from_sequence=occurrence.from_sequence + 100), *segment.occurrences[1:]),
    )
    schedule = replace(self.schedule, segments=(changed, *self.schedule.segments[1:]))

    projected = lower(schedule, device="CPU")

    self.assertEqual(projected.graph, original.graph)
    self.assertNotEqual(projected.graph_version, original.graph_version)
    self.assertEqual(projected.edges[0].occurrences[0].from_sequence, occurrence.from_sequence + 100)

  def test_missing_stale_and_canceled_state_never_becomes_observed_zero(self) -> None:
    empty = lower(self.schedule, device="CPU")
    self.assertEqual(sum(row[0] for row in empty.observed.tolist()), 0)
    self.assertEqual(sum(row[0] for row in empty.vehicle_count.tolist()), 0)

    previous = replace(self.snapshot, generated_at=self.snapshot.generated_at - 1)
    stale_trip = replace(self.snapshot.trips[1], observed_at=self.snapshot.generated_at - 91)
    trip_snapshot = replace(self.snapshot, trips=(self.snapshot.trips[0], stale_trip))
    trip_transition = evaluate(previous, trip_snapshot, self.snapshot.generated_at, POLICY)
    trip_projection = lower(self.schedule, trip_snapshot, trip_transition, device="CPU")
    self.assertEqual(sum(row[0] for row in trip_projection.observed.tolist()), 0)
    self.assertEqual(sum(row[0] for row in trip_projection.vehicle_count.tolist()), 1)

    stale_vehicle = replace(self.snapshot.vehicles[0], observed_at=self.snapshot.generated_at - 91)
    vehicle_snapshot = replace(self.snapshot, vehicles=(stale_vehicle,))
    vehicle_transition = evaluate(previous, vehicle_snapshot, self.snapshot.generated_at, POLICY)
    vehicle_projection = lower(self.schedule, vehicle_snapshot, vehicle_transition, device="CPU")
    self.assertEqual(sum(row[0] for row in vehicle_projection.observed.tolist()), 0)
    self.assertEqual(sum(row[0] for row in vehicle_projection.vehicle_count.tolist()), 0)

    canceled = replace(self.snapshot, generated_at=self.snapshot.generated_at + 1, trips=(self.snapshot.trips[0],), vehicles=())
    canceled_transition = evaluate(self.snapshot, canceled, canceled.generated_at, POLICY)
    canceled_projection = lower(self.schedule, canceled, canceled_transition, device="CPU")
    self.assertEqual(sum(row[0] for row in canceled_projection.observed.tolist()), 0)
    self.assertEqual(sum(row[0] for row in canceled_projection.vehicle_count.tolist()), 0)

  def test_rejects_unbound_or_unsupported_state(self) -> None:
    with self.assertRaisesRegex(ProjectionError, "supplied together"):
      lower(self.schedule, self.snapshot, device="CPU")
    with self.assertRaisesRegex(ProjectionError, "does not describe"):
      lower(self.schedule, replace(self.snapshot, generated_at=self.snapshot.generated_at + 1), self.transition, device="CPU")
    with self.assertRaisesRegex(ProjectionError, "does not match schedule"):
      lower(self.schedule, replace(self.snapshot, source_id="other"), replace(self.transition, current_fingerprint="invalid"), device="CPU")
    drifted = replace(self.schedule, revision="other")
    with self.assertRaisesRegex(ProjectionError, "schedule manifest"):
      lower(drifted, self.snapshot, self.transition, device="CPU")

    second = replace(self.snapshot.vehicles[0], vehicle_id="city-bus-2")
    snapshot = replace(self.snapshot, vehicles=(*self.snapshot.vehicles, second))
    key = VehicleKey(second.instance, second.vehicle_id)
    transition = replace(
      self.transition,
      current_fingerprint=fingerprint(snapshot),
      eligible_vehicles=(*self.transition.eligible_vehicles, key),
    )
    with self.assertRaisesRegex(ProjectionError, "one eligible vehicle"):
      lower(self.schedule, snapshot, transition, device="CPU")


if __name__ == "__main__":
  unittest.main()
