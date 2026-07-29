import unittest

from tinygrad import Device, Tensor

from experiments.transport_forecast import (
  FORWARD,
  LOCAL,
  REVERSE,
  GConvForecast,
  LinearDiffusionForecast,
  LocalForecast,
  _model,
  _permuted,
  _step,
  _topology,
  _trajectories,
  _validate,
)
from experiments.directed_gru import DiffusionForecast
from tinymesh import Graph
from tinymesh.nn import DirectedDiffusion


class TransportForecastTest(unittest.TestCase):
  def test_host_transport_conserves_mass_and_moves_values(self) -> None:
    topology = _topology(8)
    values = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = _step(values, topology)

    self.assertAlmostEqual(sum(result), sum(values))
    self.assertNotEqual(result, values)
    self.assertEqual((LOCAL, FORWARD, REVERSE), (0.25, 0.55, 0.20))

  def test_host_transport_matches_sparse_operator_equation(self) -> None:
    topology = _topology(8)
    values = [float(node - 4) for node in range(topology.nodes)]
    graph = Graph(topology.nodes, topology.source, topology.target)
    diffusion = DirectedDiffusion(
      graph,
      Tensor(topology.affinity, device=Device.DEFAULT).realize(),
    )
    current = Tensor([[value] for value in values], device=Device.DEFAULT)

    forward, reverse = diffusion(current)
    actual = LOCAL * current + FORWARD * forward + REVERSE * reverse

    for value, expected in zip(actual.flatten().tolist(), _step(values, topology)):
      self.assertAlmostEqual(value, expected, places=5)

  def test_trajectory_generation_is_deterministic_and_split_by_seed(self) -> None:
    topology = _topology(8)

    first = _trajectories(topology, 2, 5, 7, Device.DEFAULT)
    again = _trajectories(topology, 2, 5, 7, Device.DEFAULT)
    held_out = _trajectories(topology, 2, 5, 8, Device.DEFAULT)

    self.assertEqual(first.values.tolist(), again.values.tolist())
    self.assertNotEqual(first.values.tolist(), held_out.values.tolist())
    values, target = first.windows(3)
    self.assertEqual(values.shape, (4, 3, 8, 1))
    self.assertEqual(target.shape, (4, 8, 1))

  def test_false_topology_preserves_degree_sequence(self) -> None:
    topology = _topology(24)
    permuted = _permuted(topology)

    self.assertNotEqual(
      set(zip(topology.source, topology.target)),
      set(zip(permuted.source, permuted.target)),
    )
    for endpoint in ("source", "target"):
      original = getattr(topology, endpoint)
      false = getattr(permuted, endpoint)
      self.assertEqual(
        sorted(original.count(node) for node in range(topology.nodes)),
        sorted(false.count(node) for node in range(topology.nodes)),
      )

  def test_models_accept_batched_transport_windows(self) -> None:
    topology = _topology(8)
    values = Tensor.zeros(2, 3, 8, 1, device=Device.DEFAULT).realize()

    for name, structure, expected_type in (
      ("lstm", "none", LocalForecast),
      ("gconv_gru", "true", GConvForecast),
      ("diffusion_gru", "true", DiffusionForecast),
      ("diffusion_linear", "true", LinearDiffusionForecast),
    ):
      with self.subTest(model=name):
        model, operator, _ = _model(
          name,
          structure,
          topology,
          hidden_features=2,
          device=Device.DEFAULT,
        )
        self.assertIsInstance(model, expected_type)
        if isinstance(model, LocalForecast):
          output = model(values)
        elif isinstance(operator, Graph):
          output = model(values, operator)
        else:
          self.assertIsInstance(operator, DirectedDiffusion)
          output = model(values, operator)
        self.assertEqual(output.shape, (2, 8, 1))

  def test_invalid_model_topology_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "invalid model/topology"):
      _validate(
        "lstm",
        "true",
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
