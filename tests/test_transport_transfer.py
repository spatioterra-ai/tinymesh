import unittest

from tinygrad import Device, Tensor

from experiments.directed_gru import DiffusionForecast
from experiments.transport_forecast import (
  _model,
  _operator,
  _topology,
  _trajectories,
)
from experiments.transport_transfer import _nodes, _state, _validate
from tinymesh.nn import DirectedDiffusion


class TransportTransferTest(unittest.TestCase):
  def test_target_scope_is_explicit(self) -> None:
    self.assertEqual(_nodes("all"), (24, 32, 48))
    self.assertEqual(_nodes("32"), (32,))
    with self.assertRaisesRegex(ValueError, "target_nodes"):
      _nodes("31")

  def test_pulse_fields_are_sparse_zero_mean_and_deterministic(self) -> None:
    topology = _topology(24)

    first = _trajectories(
      topology,
      2,
      5,
      7,
      Device.DEFAULT,
      initial="pulse",
    )
    again = _trajectories(
      topology,
      2,
      5,
      7,
      Device.DEFAULT,
      initial="pulse",
    )

    self.assertEqual(first.values.tolist(), again.values.tolist())
    for field in first.values[:, 0].flatten(1).tolist():
      self.assertEqual(sum(value != 0 for value in field), 4)
      self.assertAlmostEqual(sum(field), 0.0, places=6)

  def test_one_model_accepts_unseen_graph_sizes_without_mutation(self) -> None:
    Tensor.manual_seed(0)
    model, _, _ = _model(
      "diffusion_gru",
      "true",
      _topology(24),
      hidden_features=2,
      device=Device.DEFAULT,
    )
    self.assertIsInstance(model, DiffusionForecast)
    before = _state(model)

    for nodes in (32, 48):
      topology = _topology(nodes)
      operator, _ = _operator(
        "diffusion_gru",
        "true",
        topology,
        Device.DEFAULT,
      )
      self.assertIsInstance(operator, DirectedDiffusion)

      prediction = model(
        Tensor.zeros(1, 2, nodes, 1, device=Device.DEFAULT),
        operator,
        realize_steps=True,
      )

      self.assertEqual(prediction.shape, (1, nodes, 1))
      self.assertEqual(_state(model), before)

  def test_invalid_transfer_model_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "lstm.*diffusion_gru"):
      _validate(
        "gconv_gru",
        target_nodes="all",
        initial="dense",
        seed=0,
        epochs=1,
        history=1,
        horizon=1,
        batch_size=1,
        hidden_features=1,
        learning_rate=0.01,
      )


if __name__ == "__main__":
  unittest.main()
