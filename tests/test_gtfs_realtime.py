import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from experiments.gtfs_realtime import (
  FIXTURE,
  EventPrediction,
  RealtimeError,
  normalize,
)
from experiments.gtfs_schedule import parse
from tests.test_gtfs_schedule import archive


SOURCE_ID = "google-transit/sample-feed-1"


def message() -> dict:
  return json.loads(FIXTURE.read_text())


def schedule():
  return parse(archive(), source_id=SOURCE_ID)


def active(value: dict) -> dict:
  return next(entity for entity in value["entities"] if entity["id"] == "vehicle-city1")


def canceled(value: dict) -> dict:
  return next(entity for entity in value["entities"] if entity["id"] == "cancel-city2")


class GtfsRealtimeTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.schedule = schedule()

  def test_normalizes_distinct_immutable_snapshot_facts(self) -> None:
    snapshot = normalize(message(), self.schedule, SOURCE_ID)

    self.assertEqual(snapshot.generated_at, 1167661800)
    self.assertEqual((snapshot.schedule_revision, snapshot.schedule_sha256), (self.schedule.revision, self.schedule.sha256))
    self.assertEqual([trip.entity_id for trip in snapshot.trips], ["cancel-city2", "vehicle-city1"])
    current = snapshot.trips[1]
    self.assertEqual(
      (current.instance.source_id, current.instance.trip_id, current.instance.service_date.isoformat(), current.instance.scheduled_start),
      (SOURCE_ID, "CITY1", "2007-01-01", 6 * 3600),
    )
    self.assertEqual(current.observed_at, 1167661720)
    self.assertEqual(current.delay_seconds, 120)
    self.assertEqual(current.predictions[0].arrival, EventPrediction(120, 1167662040))
    self.assertEqual(snapshot.vehicles[0].observed_at, 1167661740)
    self.assertEqual(snapshot.vehicles[0].current_stop_id, "NANAA")
    self.assertEqual(snapshot.trips[0].relationship, "CANCELED")
    self.assertIsNone(snapshot.trips[0].delay_seconds)

    with self.assertRaises(FrozenInstanceError):
      snapshot.trips[0].relationship = "SCHEDULED"

  def test_entity_permutation_preserves_semantics_and_order(self) -> None:
    original = message()
    permuted = deepcopy(original)
    permuted["entities"].reverse()

    self.assertEqual(normalize(original, self.schedule, SOURCE_ID), normalize(permuted, self.schedule, SOURCE_ID))

  def test_rejects_message_identity_and_mode_failures(self) -> None:
    cases = []

    duplicate = message()
    duplicate["entities"][1]["id"] = duplicate["entities"][0]["id"]
    cases.append(("entity id: duplicate", duplicate))

    differential = message()
    differential["header"]["incrementality"] = "DIFFERENTIAL"
    cases.append(("only FULL_DATASET", differential))

    deleted = message()
    active(deleted)["is_deleted"] = True
    cases.append(("is_deleted", deleted))

    alert = message()
    active(alert)["alert"] = {}
    cases.append(("alerts are unsupported", alert))

    for relationship in ("UNSCHEDULED", "ADDED", "NEW", "REPLACEMENT", "DUPLICATED"):
      topology = message()
      active(topology)["trip"]["schedule_relationship"] = relationship
      cases.append((f"unsupported '{relationship}'", topology))

    for reason, value in cases:
      with self.subTest(reason=reason), self.assertRaisesRegex(RealtimeError, reason):
        normalize(value, self.schedule, SOURCE_ID)

    with self.assertRaisesRegex(RealtimeError, "source_id"):
      normalize(message(), self.schedule, " ")
    with self.assertRaisesRegex(RealtimeError, "expected schedule namespace"):
      normalize(message(), self.schedule, "another-feed")

  def test_rejects_unresolved_or_inconsistent_trip_instances(self) -> None:
    cases = []

    missing_start = message()
    del active(missing_start)["trip"]["start_time"]
    cases.append(("unambiguous trip instance", missing_start))

    unknown = message()
    active(unknown)["trip"]["trip_id"] = "UNKNOWN"
    cases.append(("unresolved or ambiguous", unknown))

    route = message()
    active(route)["trip"]["route_id"] = "AB"
    cases.append(("expected 'CITY'", route))

    inactive = message()
    active(inactive)["trip"]["start_date"] = "20110101"
    cases.append(("inactive", inactive))

    start = message()
    active(start)["trip"]["start_time"] = "23:00:00"
    cases.append(("outside declared frequency", start))

    for reason, value in cases:
      with self.subTest(reason=reason), self.assertRaisesRegex(RealtimeError, reason):
        normalize(value, self.schedule, SOURCE_ID)

  def test_rejects_vehicle_clock_and_progression_failures(self) -> None:
    cases = []

    missing_time = message()
    del active(missing_time)["vehicle"]["timestamp"]
    cases.append(("timestamp: required integer", missing_time))

    future = message()
    active(future)["vehicle"]["timestamp"] = future["header"]["timestamp"] + 1
    cases.append(("newer than feed", future))

    missing_trip_time = message()
    del active(missing_trip_time)["trip_update"]["timestamp"]
    cases.append(("trip_update.timestamp: required integer", missing_trip_time))

    future_trip = message()
    active(future_trip)["trip_update"]["timestamp"] = future_trip["header"]["timestamp"] + 1
    cases.append(("trip_update.timestamp: observation.*newer than feed", future_trip))

    unknown_sequence = message()
    active(unknown_sequence)["vehicle"]["current_stop_sequence"] = 99
    cases.append(("unknown stop sequence 99", unknown_sequence))

    wrong_stop = message()
    active(wrong_stop)["vehicle"]["current_stop_id"] = "EMSI"
    cases.append(("resolves to 'NANAA'", wrong_stop))

    wrong_delay_stop = message()
    active(wrong_delay_stop)["trip_update"]["delay_at_stop_sequence"] = 1
    cases.append(("vehicle is at 2", wrong_delay_stop))

    for reason, value in cases:
      with self.subTest(reason=reason), self.assertRaisesRegex(RealtimeError, reason):
        normalize(value, self.schedule, SOURCE_ID)

  def test_rejects_invalid_stop_predictions(self) -> None:
    cases = []

    unordered = message()
    active(unordered)["trip_update"]["stop_time_updates"].reverse()
    cases.append(("unique and ordered", unordered))

    duplicate = message()
    updates = active(duplicate)["trip_update"]["stop_time_updates"]
    updates[1] = deepcopy(updates[0])
    cases.append(("unique and ordered", duplicate))

    unknown_sequence = message()
    active(unknown_sequence)["trip_update"]["stop_time_updates"][0]["stop_sequence"] = 99
    cases.append(("unknown stop sequence 99", unknown_sequence))

    wrong_stop = message()
    active(wrong_stop)["trip_update"]["stop_time_updates"][0]["stop_id"] = "EMSI"
    cases.append(("resolves to 'NADAV'", wrong_stop))

    past = message()
    active(past)["trip_update"]["stop_time_updates"][0]["stop_sequence"] = 2
    cases.append(("not future", past))

    flexible = message()
    active(flexible)["trip_update"]["stop_time_updates"][0]["location_group_id"] = "ZONE"
    cases.append(("flexible locations", flexible))

    empty_event = message()
    active(empty_event)["trip_update"]["stop_time_updates"][0]["arrival"] = {}
    cases.append(("delay or time is required", empty_event))

    reversed_event = message()
    first = active(reversed_event)["trip_update"]["stop_time_updates"][0]
    first["departure"]["time"] = first["arrival"]["time"] - 1
    cases.append(("precedes arrival", reversed_event))

    for reason, value in cases:
      with self.subTest(reason=reason), self.assertRaisesRegex(RealtimeError, reason):
        normalize(value, self.schedule, SOURCE_ID)

  def test_rejects_contradictory_state(self) -> None:
    canceled_active = message()
    canceled(canceled_active)["vehicle"] = deepcopy(active(canceled_active)["vehicle"])
    with self.assertRaisesRegex(RealtimeError, "canceled trip contradicts"):
      normalize(canceled_active, self.schedule, SOURCE_ID)

    relationships = message()
    contradiction = deepcopy(canceled(relationships))
    contradiction["id"] = "cancel-city1"
    contradiction["trip"] = deepcopy(active(relationships)["trip"])
    contradiction["trip"]["schedule_relationship"] = "CANCELED"
    relationships["entities"].append(contradiction)
    with self.assertRaisesRegex(RealtimeError, "contradictory relationships"):
      normalize(relationships, self.schedule, SOURCE_ID)

    duplicate = message()
    repeated = deepcopy(active(duplicate))
    repeated["id"] = "vehicle-city1-copy"
    repeated["vehicle"]["vehicle_id"] = "city-bus-2"
    duplicate["entities"].append(repeated)
    with self.assertRaisesRegex(RealtimeError, "trip instance: duplicate"):
      normalize(duplicate, self.schedule, SOURCE_ID)


if __name__ == "__main__":
  unittest.main()
