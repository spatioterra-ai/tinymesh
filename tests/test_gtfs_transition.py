import json
import unittest
from dataclasses import asdict, replace

from experiments.gtfs_realtime import FIXTURE, normalize
from experiments.gtfs_schedule import parse
from experiments.gtfs_transition import (
  ActiveAfterTerminal,
  BLOCKING,
  ContentChangedWithoutGeneration,
  FeedGapExceeded,
  FeedGenerationRegressed,
  POLICY,
  TransitionError,
  TransitionPolicy,
  TripObservationStale,
  VehicleObservationStale,
  VehicleStopRegressed,
  evaluate,
  fingerprint,
)
from tests.test_gtfs_schedule import archive


SOURCE_ID = "google-transit/sample-feed-1"
POLICY_30_90 = TransitionPolicy(30, 90, 90)


def codes(transition) -> list[str]:
  return [finding.code for finding in transition.findings]


class GtfsTransitionTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    schedule = parse(archive(), source_id=SOURCE_ID)
    cls.snapshot = normalize(json.loads(FIXTURE.read_text()), schedule, SOURCE_ID)

  def test_accepts_forward_generation_and_vehicle_progression_deterministically(self) -> None:
    current = self.snapshot
    prior_vehicle = replace(current.vehicles[0], current_stop_id="STAGECOACH", current_stop_sequence=1)
    previous = replace(current, generated_at=current.generated_at - 10, vehicles=(prior_vehicle,))

    first = evaluate(previous, current, current.generated_at, POLICY_30_90)
    second = evaluate(previous, current, current.generated_at, POLICY_30_90)

    self.assertEqual(first, second)
    self.assertEqual(first.findings, ())
    self.assertEqual(len(first.eligible_trips), 2)
    self.assertEqual(len(first.eligible_vehicles), 1)

  def test_fingerprint_is_order_invariant_and_semantic(self) -> None:
    snapshot = self.snapshot
    permuted = replace(snapshot, trips=tuple(reversed(snapshot.trips)), vehicles=tuple(reversed(snapshot.vehicles)))
    changed_trip = replace(snapshot.trips[1], delay_seconds=snapshot.trips[1].delay_seconds + 1)
    changed = replace(snapshot, trips=(snapshot.trips[0], changed_trip))

    self.assertEqual(fingerprint(snapshot), fingerprint(permuted))
    self.assertNotEqual(fingerprint(snapshot), fingerprint(changed))

  def test_allows_unchanged_content_under_later_generation(self) -> None:
    current = replace(self.snapshot, generated_at=self.snapshot.generated_at + 1)

    transition = evaluate(self.snapshot, current, current.generated_at, POLICY_30_90)

    self.assertEqual(transition.findings, ())

  def test_blocks_changed_content_without_generation_progress(self) -> None:
    active = replace(self.snapshot.trips[1], delay_seconds=121)
    current = replace(self.snapshot, trips=(self.snapshot.trips[0], active))

    transition = evaluate(self.snapshot, current, current.generated_at, POLICY_30_90)

    self.assertEqual(codes(transition), ["CONTENT_CHANGED_WITHOUT_GENERATION"])
    finding = transition.findings[0]
    self.assertIsInstance(finding, ContentChangedWithoutGeneration)
    self.assertEqual((finding.severity, finding.previous_at, finding.current_at), (BLOCKING, current.generated_at, current.generated_at))
    self.assertEqual((transition.eligible_trips, transition.eligible_vehicles), ((), ()))

  def test_blocks_decreasing_generation(self) -> None:
    current = replace(self.snapshot, generated_at=self.snapshot.generated_at - 1)

    transition = evaluate(self.snapshot, current, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(transition), ["FEED_GENERATION_REGRESSED"])
    self.assertIsInstance(transition.findings[0], FeedGenerationRegressed)
    self.assertEqual(transition.findings[0].severity, BLOCKING)
    self.assertEqual((transition.eligible_trips, transition.eligible_vehicles), ((), ()))

  def test_reports_feed_gap_without_silently_rejecting_current_facts(self) -> None:
    previous = replace(self.snapshot, generated_at=self.snapshot.generated_at - 31)

    transition = evaluate(previous, self.snapshot, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(transition), ["FEED_GAP_EXCEEDED"])
    finding = transition.findings[0]
    self.assertIsInstance(finding, FeedGapExceeded)
    self.assertEqual((finding.severity, finding.limit_seconds), (POLICY, 30))
    self.assertEqual(len(transition.eligible_trips), 2)
    self.assertEqual(len(transition.eligible_vehicles), 1)

  def test_stale_trip_and_vehicle_facts_are_independently_ineligible(self) -> None:
    previous = replace(self.snapshot, generated_at=self.snapshot.generated_at - 1)
    stale_trip = replace(self.snapshot.trips[1], observed_at=self.snapshot.generated_at - 91)
    trip_current = replace(self.snapshot, trips=(self.snapshot.trips[0], stale_trip))

    trip_transition = evaluate(previous, trip_current, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(trip_transition), ["TRIP_OBSERVATION_STALE"])
    self.assertIsInstance(trip_transition.findings[0], TripObservationStale)
    self.assertEqual(trip_transition.findings[0].severity, POLICY)
    self.assertNotIn(stale_trip.instance, trip_transition.eligible_trips)
    self.assertEqual(len(trip_transition.eligible_vehicles), 1)

    stale_vehicle = replace(self.snapshot.vehicles[0], observed_at=self.snapshot.generated_at - 91)
    vehicle_current = replace(self.snapshot, vehicles=(stale_vehicle,))
    vehicle_transition = evaluate(previous, vehicle_current, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(vehicle_transition), ["VEHICLE_OBSERVATION_STALE"])
    self.assertIsInstance(vehicle_transition.findings[0], VehicleObservationStale)
    self.assertEqual(len(vehicle_transition.eligible_trips), 2)
    self.assertEqual(vehicle_transition.eligible_vehicles, ())

  def test_blocks_backward_stop_progression_within_one_instance(self) -> None:
    prior_vehicle = replace(self.snapshot.vehicles[0], current_stop_id="NADAV", current_stop_sequence=3)
    previous = replace(self.snapshot, generated_at=self.snapshot.generated_at - 1, vehicles=(prior_vehicle,))

    transition = evaluate(previous, self.snapshot, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(transition), ["VEHICLE_STOP_REGRESSED"])
    finding = transition.findings[0]
    self.assertIsInstance(finding, VehicleStopRegressed)
    self.assertEqual(
      (finding.vehicle_id, finding.previous_stop_sequence, finding.current_stop_sequence),
      ("city-bus-1", 3, 2),
    )
    self.assertEqual((finding.previous_at, finding.current_at), (None, None))
    self.assertEqual(
      asdict(finding),
      {
        "code": "VEHICLE_STOP_REGRESSED",
        "severity": "blocking",
        "source_id": SOURCE_ID,
        "instance": asdict(finding.instance),
        "vehicle_id": "city-bus-1",
        "previous_stop_sequence": 3,
        "current_stop_sequence": 2,
        "previous_at": None,
        "current_at": None,
        "as_of": self.snapshot.generated_at,
        "limit_seconds": None,
      },
    )
    self.assertEqual(len(transition.eligible_trips), 2)
    self.assertEqual(transition.eligible_vehicles, ())

  def test_new_trip_instance_resets_vehicle_progression(self) -> None:
    prior_vehicle = replace(self.snapshot.vehicles[0], current_stop_id="DADAN", current_stop_sequence=4)
    previous = replace(self.snapshot, generated_at=self.snapshot.generated_at - 1, vehicles=(prior_vehicle,))
    instance = replace(self.snapshot.trips[1].instance, scheduled_start=self.snapshot.trips[1].instance.scheduled_start + 1800)
    trip = replace(self.snapshot.trips[1], instance=instance)
    vehicle = replace(self.snapshot.vehicles[0], instance=instance, current_stop_id="STAGECOACH", current_stop_sequence=1)
    current = replace(self.snapshot, generated_at=self.snapshot.generated_at + 1, trips=(self.snapshot.trips[0], trip), vehicles=(vehicle,))

    transition = evaluate(previous, current, current.generated_at, POLICY_30_90)

    self.assertNotIn("VEHICLE_STOP_REGRESSED", codes(transition))
    self.assertEqual(len(transition.eligible_vehicles), 1)

  def test_blocks_active_state_after_terminal_trip(self) -> None:
    terminal = replace(self.snapshot.trips[1], relationship="CANCELED", delay_seconds=None, predictions=())
    cancellation = replace(self.snapshot.trips[0], observed_at=self.snapshot.generated_at - 1)
    previous = replace(
      self.snapshot,
      generated_at=self.snapshot.generated_at - 1,
      trips=(cancellation, terminal),
      vehicles=(),
    )

    transition = evaluate(previous, self.snapshot, self.snapshot.generated_at, POLICY_30_90)

    self.assertEqual(codes(transition), ["ACTIVE_AFTER_TERMINAL"])
    self.assertIsInstance(transition.findings[0], ActiveAfterTerminal)
    self.assertNotIn(self.snapshot.trips[1].instance, transition.eligible_trips)
    self.assertEqual(transition.eligible_vehicles, ())

  def test_absence_from_full_snapshot_remains_unknown(self) -> None:
    current = replace(
      self.snapshot,
      generated_at=self.snapshot.generated_at + 1,
      trips=(self.snapshot.trips[0],),
      vehicles=(),
    )

    transition = evaluate(self.snapshot, current, current.generated_at, POLICY_30_90)

    self.assertEqual(transition.findings, ())
    self.assertEqual(transition.eligible_trips, (self.snapshot.trips[0].instance,))
    self.assertEqual(transition.eligible_vehicles, ())

  def test_rejects_invalid_evaluation_context(self) -> None:
    with self.assertRaisesRegex(TransitionError, "source mismatch"):
      evaluate(self.snapshot, replace(self.snapshot, source_id="other"), self.snapshot.generated_at, POLICY_30_90)
    with self.assertRaisesRegex(TransitionError, "as_of"):
      evaluate(self.snapshot, self.snapshot, self.snapshot.generated_at - 1, POLICY_30_90)
    with self.assertRaisesRegex(TransitionError, "non-negative"):
      evaluate(self.snapshot, self.snapshot, self.snapshot.generated_at, TransitionPolicy(-1, 90, 90))

  def test_finding_variants_require_their_subject(self) -> None:
    with self.assertRaisesRegex(TypeError, "instance"):
      ActiveAfterTerminal(source_id=SOURCE_ID, as_of=self.snapshot.generated_at)


if __name__ == "__main__":
  unittest.main()
