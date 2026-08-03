import unittest

from tinygrad import Context, Device, Tensor, nn

from experiments.metr_la_forecast import _execution_batch, _operators
from experiments.metr_la_jepa import (
    ARMS,
    FactorizedEncoder,
    Model,
    _blocks,
    _arm,
    _persistence,
    _snapshot,
    _sparse_calls,
    _update_target,
    _validate,
    _variation,
)
from experiments.metr_la_protocol import batches, prepare
from tests.test_metr_la_forecast import dataset


class METRLAJEPATest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = prepare(dataset(steps=60), history=2, horizon=3, feature_set="calendar")
        self.operators = _operators(self.protocol, "local_diffusion", Device.DEFAULT)

    def test_factorized_encoder_preserves_time_and_node_axes(self) -> None:
        values = Tensor.randn(3, 2, 4, 2, device=Device.DEFAULT).realize()
        encoder = FactorizedEncoder(2, 3, 2)
        output = encoder(values, self.operators["true"])
        zero = encoder(Tensor.zeros_like(values), self.operators["true"])
        Tensor.realize(output, zero)

        self.assertEqual(output.shape, (3, 2, 4, 3))
        self.assertEqual(zero.abs().max().item(), 0.0)

    def test_self_topology_keeps_temporal_mixing_node_local(self) -> None:
        values = Tensor.randn(1, 2, 4, 2, device=Device.DEFAULT).realize()
        changed = values.clone().realize()
        changed[:, :, 0] = changed[:, :, 0] + 10
        encoder = FactorizedEncoder(2, 3, 2)

        original = encoder(values, self.operators["self"])
        perturbed = encoder(changed, self.operators["self"])

        self.assertLess((original[:, :, 1:] - perturbed[:, :, 1:]).abs().max().item(), 1e-6)

    def test_topology_arms_start_from_one_node_local_representation(self) -> None:
        values = Tensor.randn(2, 2, 4, 2, device=Device.DEFAULT).realize()
        encoder = FactorizedEncoder(2, 3, 2)

        true = encoder(values, self.operators["true"])
        permuted = encoder(values, self.operators["permuted"])
        temporal = encoder(values, self.operators["self"])

        self.assertLess((true - permuted).abs().max().item(), 1e-6)
        self.assertLess((true - temporal).abs().max().item(), 1e-6)

        encoder.spatial.gate.assign(Tensor.ones_like(encoder.spatial.gate)).realize()
        self.assertGreater(
            (encoder(values, self.operators["true"]) - encoder(values, self.operators["self"])).abs().max().item(),
            0.0,
        )

    def test_variation_measures_examples_not_static_nodes(self) -> None:
        static = Tensor.arange(12).reshape(1, 4, 3).expand(5, 4, 3)
        varying = static + Tensor.arange(5).reshape(5, 1, 1)

        self.assertEqual(_variation(static), 0.0)
        self.assertGreater(_variation(varying), 0.0)

    def test_target_stops_gradient_and_follows_online_encoder(self) -> None:
        batch = _execution_batch(
            next(batches(self.protocol, self.protocol.train, batch_size=4)),
            Device.DEFAULT,
            4,
        )
        context, target = _blocks(batch)
        Tensor.manual_seed(3)
        model = Model(2, 3, 2, 3, temporal=True)
        _update_target(model.online, model.target, 0)
        initial = _snapshot(model.target)

        with Context(TRAINING=1):
            model.loss(context, target, self.operators["true"]).backward()
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
        self.assertEqual(_sparse_calls(model, context, target, self.operators["true"]), 4)

    def test_arms_assign_the_matched_controls(self) -> None:
        expected = {
            "factorized": ("true", True),
            "permuted": ("permuted", True),
            "temporal": ("self", True),
            "spatial": ("true", False),
        }
        self.assertEqual(tuple(expected), ARMS)
        for name, (topology, temporal) in expected.items():
            actual_topology, actual_temporal, diffusion = _arm(name, self.operators)
            self.assertEqual((actual_topology, actual_temporal), (topology, temporal))
            self.assertIs(diffusion, self.operators[topology])
        with self.assertRaises(ValueError):
            _arm("unknown", self.operators)

    def test_persistence_uses_the_fixed_sample(self) -> None:
        tensors = (self.protocol.features[..., :2], self.protocol.target, self.protocol.observed)
        batch = next(batches(self.protocol, self.protocol.validation, 4, shuffle=7, tensors=tensors))
        result = _persistence(self.protocol, self.protocol.validation, tensors, 4, 4, 7)

        self.assertEqual(result.overall.observations, int(batch.observed.sum().item()))

    def test_validation_rejects_invalid_protocol(self) -> None:
        valid = dict(
            seed=0,
            steps=1,
            probe_steps=1,
            probe_samples=8,
            evaluation_samples=4,
            batch_size=4,
            hidden_features=3,
            learning_rate=0.01,
            ema_decay=0.99,
            probe_learning_rate=0.01,
            evaluate_test=False,
        )
        for name, value in (
            ("seed", -1),
            ("steps", 0),
            ("probe_samples", 7),
            ("evaluation_samples", 8),
            ("batch_size", 9),
            ("hidden_features", 1),
            ("learning_rate", 0),
            ("ema_decay", 1),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate(self.protocol, **(valid | {name: value}))


if __name__ == "__main__":
    unittest.main()
