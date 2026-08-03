import unittest

from tinygrad import Device, Tensor, nn

from experiments.transport_forecast import _topology, _trajectories
from experiments.transport_jepa import Model, MeshEncoder, _mesh, _snapshot, _standardize, _update_target, _validate


class TransportJEPATest(unittest.TestCase):
  def setUp(self) -> None:
    self.topology = _topology(24)

  def test_product_mesh_separates_time_and_directed_space(self) -> None:
    meshes = {name: _mesh(self.topology, name, 4, Device.DEFAULT) for name in ("true", "permuted", "temporal", "spatial")}

    self.assertEqual((meshes["true"].graph.nodes, meshes["true"].graph.edges), (96, 360))
    self.assertEqual((meshes["true"].temporal_edges, meshes["true"].spatial_edges), (72, 288))
    self.assertEqual((meshes["temporal"].temporal_edges, meshes["temporal"].spatial_edges), (72, 0))
    self.assertEqual((meshes["spatial"].temporal_edges, meshes["spatial"].spatial_edges), (0, 288))
    self.assertNotEqual(meshes["true"].graph, meshes["permuted"].graph)
    self.assertEqual(meshes["true"].edge_values[:72].tolist(), [[1.0, 0.0, 0.0]] * 72)
    self.assertTrue(all(source // 24 + 1 == target // 24 and source % 24 == target % 24 for source, target in zip(
      meshes["true"].graph.source[:72], meshes["true"].graph.target[:72],
    )))
    self.assertTrue(all(source // 24 == target // 24 for source, target in zip(
      meshes["true"].graph.source[72:], meshes["true"].graph.target[72:],
    )))

  def test_encoder_preserves_batch_and_node_identity(self) -> None:
    mesh = _mesh(self.topology, "true", 4, Device.DEFAULT)
    values = Tensor.randn(2, 4, 24, 1, device=Device.DEFAULT).realize()
    encoder = MeshEncoder(4)
    output = encoder(values, mesh)
    zero = encoder(Tensor.zeros_like(values), mesh)
    gradient = output.sum().gradient(values)[0]
    Tensor.realize(output, zero, gradient)

    self.assertEqual(output.shape, (2, 4, 24, 4))
    self.assertEqual(zero.abs().max().item(), 0)
    self.assertEqual(gradient.shape, values.shape)

  def test_target_stops_gradient_and_follows_online_encoder(self) -> None:
    data = _trajectories(self.topology, 4, 4, 7, Device.DEFAULT)
    context, target = data.values[:, :2].contiguous().realize(), data.values[:, 2:].contiguous().realize()
    mesh = _mesh(self.topology, "true", 2, Device.DEFAULT)
    Tensor.manual_seed(3)
    model = Model(4)
    _update_target(model.online, model.target, 0)
    initial = _snapshot(model.target)
    self.assertEqual(
      model.online(context, mesh).tolist(),
      model.target(context, mesh).tolist(),
    )

    model.loss(context, target, mesh, mesh).backward()
    online_gradient = sum(
      parameter.grad.abs().sum().item()
      for parameter in nn.state.get_parameters(model.online)
      if parameter.grad is not None
    )
    target_gradient = sum(
      0 if parameter.grad is None else parameter.grad.abs().sum().item()
      for parameter in nn.state.get_parameters(model.target)
    )
    self.assertGreater(online_gradient, 0)
    self.assertEqual(target_gradient, 0)

    parameter = nn.state.get_parameters(model.online)[0]
    parameter.assign(parameter + 0.1).realize()
    _update_target(model.online, model.target, 0.5)
    self.assertGreater(sum(
      (value - initial[name]).abs().sum().item()
      for name, value in nn.state.get_state_dict(model.target).items()
    ), 0)

  def test_probe_standardization_uses_training_rows(self) -> None:
    train = Tensor([[[1.0, 4.0]], [[3.0, 8.0]]], device=Device.DEFAULT).realize()
    validation = Tensor([[[5.0, 12.0]]], device=Device.DEFAULT).realize()
    test = Tensor([[[7.0, 16.0]]], device=Device.DEFAULT).realize()
    normalized, normalized_validation, normalized_test = _standardize(train, validation, test)

    self.assertLess(normalized.mean().abs().item(), 1e-6)
    self.assertLess((normalized_validation - 2.12132034).abs().max().item(), 1e-6)
    self.assertLess((normalized_test - 3.53553391).abs().max().item(), 1e-6)

  def test_validation_rejects_invalid_protocol(self) -> None:
    valid = dict(
      seed=0,
      history=4,
      horizon=4,
      steps=1,
      probe_steps=1,
      hidden_features=4,
      learning_rate=0.01,
      ema_decay=0.99,
      probe_learning_rate=0.01,
    )
    for name, value in (("seed", -1), ("history", 0), ("horizon", 3), ("hidden_features", 1), ("learning_rate", 0), ("ema_decay", 1)):
      with self.subTest(name=name), self.assertRaises(ValueError):
        _validate(**(valid | {name: value}))


if __name__ == "__main__":
  unittest.main()
