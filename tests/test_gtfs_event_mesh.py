import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.gtfs_event_mesh import EventMesh, EventMeshError, Relation, lower, observe
from experiments.gtfs_replay import FIXTURE, RetainedReplay, TripInstance, load


class GtfsEventMeshTest(unittest.TestCase):
  def test_exact_mesh_and_clock_audit(self) -> None:
    result = observe()

    self.assertEqual(
      (
        result.mesh.source_rows,
        result.mesh.represented_source_rows,
        result.mesh.physical_departures,
        result.mesh.source_aliases,
        result.mesh.run_relations,
        result.mesh.headway_relations,
      ),
      (663, 597, 597, 597, 534, 575),
    )
    self.assertEqual(
      (
        result.mesh.source_headways,
        result.mesh.exact_headways,
        result.mesh.derived_only_headways,
        result.mesh.boundary_only_headways,
        result.mesh.max_in_degree,
        result.mesh.max_out_degree,
        result.mesh.event_records,
        result.mesh.relation_records,
        result.mesh.sparse_records,
        result.mesh.dense_event_pair_cells,
      ),
      (608, 575, 0, 33, 2, 2, 597, 1109, 1706, 356409),
    )
    self.assertEqual(
      (
        result.midpoint_prefix.retained_events,
        result.midpoint_prefix.retained_relations,
        result.midpoint_prefix.excluded_events,
        result.midpoint_prefix.crossing_relations,
      ),
      (297, 536, 300, 30),
    )
    self.assertEqual(
      [
        (
          clock.seconds,
          clock.cells,
          clock.occupied_cells,
          clock.empty_cells,
          clock.colliding_cells,
          clock.merged_events,
          clock.max_events_per_cell,
        )
        for clock in result.clocks
      ],
      [
        (30, 5280, 575, 4705, 0, 0, 1),
        (60, 2640, 575, 2065, 0, 0, 1),
        (300, 528, 485, 43, 90, 90, 2),
      ],
    )
    self.assertEqual(result.carrier_decision, "retain:event_mesh")

  def test_event_aliases_round_trip_to_source_and_schedule(self) -> None:
    source = load()
    mesh = lower(source)
    rows = {(row.instance, row.stop_sequence): row for row in source.rows}
    calls = {(call.trip_id, call.stop_sequence): call for call in source.calls}

    for event in mesh.events:
      for alias in event.aliases:
        row = rows[alias.source_identity]
        call = calls[(alias.instance.trip_id, alias.stop_sequence)]
        following = calls[(alias.instance.trip_id, alias.following_stop_sequence)]
        self.assertEqual((row.stop_id, row.parent_station), (alias.stop_id, event.key.parent_station))
        self.assertEqual((call.stop_id, call.departure_time), (alias.stop_id, alias.scheduled_departure_time))
        self.assertEqual(following.stop_id, alias.following_stop_id)

  def test_relations_are_typed_forward_and_deterministic(self) -> None:
    source = load()
    first = lower(source)
    second = lower(source)

    self.assertEqual(first, second)
    self.assertEqual({relation.kind for relation in first.relations}, {"headway", "run"})
    for relation in first.relations:
      self.assertLess(relation.source.departure_timestamp, relation.target.departure_timestamp)
      self.assertEqual(
        relation.elapsed_seconds,
        relation.target.departure_timestamp - relation.source.departure_timestamp,
      )

  def test_alias_collapse_preserves_one_physical_event(self) -> None:
    source, aliased = self._duplicate_trip()
    original = lower(source)
    mesh = lower(aliased)

    self.assertEqual(tuple(event.key for event in mesh.events), tuple(event.key for event in original.events))
    self.assertGreater(sum(len(event.aliases) - 1 for event in mesh.events), 0)
    self.assertEqual(mesh.relations, original.relations)

  def test_conflicting_aliases_fail_with_physical_identity(self) -> None:
    _, aliased = self._duplicate_trip()
    alias_instance = next(row.instance for row in aliased.rows if row.instance.trip_id.endswith("-alias"))
    rows = list(aliased.rows)
    index = next(
      index
      for index, row in enumerate(rows)
      if row.instance == alias_instance and row.headway_trunk_seconds is not None
    )
    headway = rows[index].headway_trunk_seconds
    assert headway is not None
    rows[index] = replace(rows[index], headway_trunk_seconds=headway + 1)

    with self.assertRaisesRegex(EventMeshError, "alias: conflicting physical departure"):
      lower(replace(aliased, rows=tuple(rows)))

  def test_non_forward_relation_fails_with_endpoint_identity(self) -> None:
    source = load()
    instance = next(instance for instance in {row.instance for row in source.rows} if sum(row.instance == instance for row in source.rows) >= 3)
    trip_rows = sorted((row for row in source.rows if row.instance == instance), key=lambda row: row.stop_sequence)
    self.assertIsNotNone(trip_rows[1].move_timestamp)
    rows = tuple(
      replace(row, move_timestamp=trip_rows[1].move_timestamp) if row == trip_rows[2] else row
      for row in source.rows
    )

    with self.assertRaisesRegex(EventMeshError, "relation: time did not advance"):
      lower(replace(source, rows=rows))

  def test_prefix_excludes_target_and_incident_relations(self) -> None:
    mesh = lower(load())
    target = next(relation.target for relation in mesh.relations if relation.kind == "headway")
    prefix = mesh.prefix(target.departure_timestamp)
    retained = {event.key for event in prefix.events}

    self.assertTrue(all(event.key.departure_timestamp < target.departure_timestamp for event in prefix.events))
    self.assertNotIn(target, retained)
    self.assertTrue(all(relation.source in retained and relation.target in retained for relation in prefix.relations))

  def test_relation_validation_rejects_self_duplicate_contradiction_and_unknown(self) -> None:
    mesh = lower(load())
    relation = mesh.relations[0]
    source = next(event for event in mesh.events if event.key == relation.source)

    with self.assertRaisesRegex(EventMeshError, "self edge"):
      EventMesh((source,), (Relation("run", source.key, source.key, 0),))
    with self.assertRaisesRegex(EventMeshError, "duplicate"):
      EventMesh(mesh.events, (relation, relation))
    other = "run" if relation.kind == "headway" else "headway"
    with self.assertRaisesRegex(EventMeshError, "contradictory kinds"):
      EventMesh(mesh.events, (relation, replace(relation, kind=other)))
    unknown = replace(relation.source, vehicle_id="unknown")
    with self.assertRaisesRegex(EventMeshError, "unresolved endpoint"):
      EventMesh(mesh.events, (replace(relation, source=unknown),))

  def test_checksum_drift_fails_before_lowering(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      replay = fixture / "replay.csv"
      replay.write_bytes(replay.read_bytes() + b"\n")

      with self.assertRaisesRegex(ValueError, "does not match retained artifact manifest"):
        observe(fixture)

  @staticmethod
  def _duplicate_trip() -> tuple[RetainedReplay, RetainedReplay]:
    source = load()
    instance = next(
      instance
      for instance in sorted({row.instance for row in source.rows})
      if sum(row.instance == instance for row in source.rows) >= 2
    )
    alias_instance = TripInstance(instance.service_date, instance.start_time, f"{instance.trip_id}-alias")
    alias_rows = tuple(replace(row, instance=alias_instance) for row in source.rows if row.instance == instance)
    alias_calls = tuple(
      replace(call, trip_id=alias_instance.trip_id)
      for call in source.calls
      if call.trip_id == instance.trip_id
    )
    return source, replace(source, rows=source.rows + alias_rows, calls=source.calls + alias_calls)


if __name__ == "__main__":
  unittest.main()
