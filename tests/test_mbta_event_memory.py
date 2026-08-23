import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor

from experiments.mbta_event_memory import (
  Event,
  EventEncoder,
  EventMemoryError,
  Memory,
  Query,
  observe,
  replay,
  replay_at,
)
from experiments.tools import mbta_event_memory as tool


class MbtaEventMemoryTest(unittest.TestCase):
  def test_retained_validation_stops_before_test(self) -> None:
    result = observe()

    self.assertEqual((result.targets, result.split, result.decision), (138_910, "validation", "stop:event_memory"))
    self.assertEqual(result.true_mae_seconds, 147.725922)
    self.assertLess(result.true_mae_seconds, result.permuted_mae_seconds)

  def test_prediction_reads_before_event_update(self) -> None:
    reads = replay((Event(10, 0, 0, 3.0), Event(20, 1, 0, 5.0)), nodes=1)

    self.assertEqual((reads[0].value, reads[0].elapsed), (None, None))
    self.assertEqual((reads[1].value, reads[1].elapsed), (3.0, 10))

  def test_equal_time_events_share_old_state_then_aggregate(self) -> None:
    events = (
      Event(10, 0, 0, 2.0),
      Event(20, 1, 0, 4.0),
      Event(20, 2, 0, 8.0),
      Event(30, 3, 0, 10.0),
    )
    reads = replay(events, nodes=1)

    self.assertEqual((reads[1].value, reads[2].value), (2.0, 2.0))
    self.assertEqual((reads[1].elapsed, reads[2].elapsed), (10, 10))
    self.assertEqual((reads[3].value, reads[3].elapsed), (6.0, 10))

  def test_replay_resets_and_is_deterministic(self) -> None:
    events = (Event(20, 1, 1, 5.0), Event(10, 0, 0, 3.0))

    self.assertEqual(replay(events, 2), replay(tuple(reversed(events)), 2))
    self.assertIsNone(replay((Event(30, 2, 0, 7.0),), 2)[0].value)

  def test_frozen_cutoff_does_not_reveal_future_departure_time(self) -> None:
    events = (Event(10, 0, 0, 3.0), Event(20, 1, 0, 5.0))
    reads = replay_at(events, (Query(11, 0, 0), Query(21, 1, 0)), nodes=1)

    self.assertEqual((reads[0].value, reads[0].elapsed), (3.0, 1))
    self.assertEqual((reads[1].value, reads[1].elapsed), (5.0, 1))

  def test_memory_rejects_invalid_lifecycle(self) -> None:
    memory = Memory.empty(1).advance((Event(10, 0, 0, 1.0),))
    with self.assertRaisesRegex(EventMemoryError, "strictly after"):
      memory.read(Event(10, 1, 0, 2.0))
    with self.assertRaisesRegex(EventMemoryError, "mixes timestamps"):
      memory.advance((Event(20, 1, 0, 2.0), Event(21, 2, 0, 3.0)))
    with self.assertRaisesRegex(EventMemoryError, "out of range"):
      memory.read(Event(20, 1, 1, 2.0))

  def test_encoder_preserves_bounded_state_shape(self) -> None:
    Tensor.manual_seed(0)
    encoder = EventEncoder(hidden=4, time_features=2)
    state = Tensor.zeros(3, 4)
    elapsed = Tensor([[0.1], [0.2], [0.3]])

    updated = encoder.update(elapsed, state)

    self.assertEqual(updated.shape, (3, 4))
    self.assertEqual(encoder.predict(updated, Tensor.zeros(3)).shape, (3,))
    with self.assertRaisesRegex(EventMemoryError, "shape drift"):
      encoder.update(Tensor.zeros(3, 2), state)

  def test_event_identity_is_unique(self) -> None:
    with self.assertRaisesRegex(EventMemoryError, "duplicate identity"):
      replay((Event(10, 0, 0, 1.0), Event(20, 0, 0, 2.0)), 1)

  def test_artifact_drift_fails_before_source_rebuild(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      arguments = {
        "source_manifest_sha256": source / "manifest.json",
        "population_audit_sha256": root / "population.json",
        "task_protocol_sha256": root / "task.json",
        "topology_protocol_sha256": root / "topology.json",
        "topology_validation_sha256": root / "topology-validation.json",
        "topology_test_sha256": root / "topology-test.json",
        "clock_audit_sha256": root / "clock.json",
      }
      for name, path in arguments.items():
        path.write_text(name)
      expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in arguments.items()
      }
      arguments["topology_protocol_sha256"].write_text("drift")
      with patch.object(tool, "EXPECTED_ARTIFACTS", expected), patch.object(tool, "build_topology") as rebuild:
        with self.assertRaisesRegex(tool.ExperimentError, "frozen digest drift"):
          tool.build(source, *(arguments[name] for name in tuple(arguments)[1:]))
      rebuild.assert_not_called()


if __name__ == "__main__":
  unittest.main()
