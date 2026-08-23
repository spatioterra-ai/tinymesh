import unittest
from datetime import datetime, timedelta
from math import isfinite

from tinygrad import Context, Device, Tensor, nn

from experiments.metr_la_forecast import (
    A3Forecast,
    DiffusionForecast,
    LocalDiffusionForecast,
    _objective,
    forecast,
    train,
)
from experiments.metr_la_protocol import _baseline, baselines, batches, execution_batch, graphs, operators, prepare
from tinymesh import Graph
from tinymesh.datasets import METRLA


def dataset(*, future_shift: float = 0.0, steps: int = 30) -> METRLA:
    graph = Graph(
        4,
        [0, 1, 2, 3, 0, 1, 2, 2],
        [0, 1, 2, 3, 1, 2, 0, 3],
    )
    rows = []
    for time in range(steps):
        shift = future_shift if time >= 21 else 0.0
        rows.append([
            10.0 + time + shift,
            20.0 + 2 * time + shift,
            30.0 + time % 3 + shift,
            40.0 + shift,
        ])
    rows[1][1] = 0.0
    rows[20][0] = 0.0
    rows[22][2] = 0.0
    speed = Tensor(rows, device=Device.DEFAULT).realize()
    return METRLA(
        graph,
        ("a", "b", "c", "d"),
        tuple(datetime(2012, 3, 1) + timedelta(minutes=5 * time) for time in range(steps)),
        speed,
        Tensor.ones(graph.edges, dtype=speed.dtype, device=speed.device).realize(),
    )


class METRLAForecastTest(unittest.TestCase):
    def test_target_time_windows_do_not_cross_split_targets(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3)

        self.assertEqual((protocol.train_end, protocol.validation_end), (21, 24))
        self.assertEqual(
            (protocol.train.windows, protocol.validation.windows, protocol.test.windows),
            (17, 1, 4),
        )
        validation = next(batches(protocol, protocol.validation, batch_size=8))
        self.assertEqual(validation.starts, (21,))
        self.assertEqual(
            validation.values.shape,
            (1, 2, 4, 2),
        )
        self.assertEqual(validation.observed[0, 2].tolist(), [True, False, True])

    def test_preprocessing_uses_observed_training_rows_only(self) -> None:
        original = prepare(dataset(), history=2, horizon=3)
        changed = prepare(dataset(future_shift=1000), history=2, horizon=3)

        self.assertEqual(original.standardizer.mean.tolist(), changed.standardizer.mean.tolist())
        self.assertEqual(original.standardizer.scale.tolist(), changed.standardizer.scale.tolist())
        self.assertEqual(original.node_mean.tolist(), changed.node_mean.tolist())
        self.assertEqual(original.seasonal_mean.tolist(), changed.seasonal_mean.tolist())
        self.assertEqual(original.target[1, 1].item(), 0.0)
        self.assertEqual(original.features[1, 1, 0].item(), 0.0)

    def test_calendar_features_are_causal_cycles_with_explicit_missingness(self) -> None:
        protocol = prepare(dataset(steps=289), history=2, horizon=3, feature_set="calendar")

        self.assertEqual(
            protocol.feature_names,
            ("speed", "observed", "sin_day", "cos_day", "sin_week", "cos_week"),
        )
        self.assertEqual(protocol.features.shape, (289, 4, 6))
        self.assertEqual(protocol.feature_statistics, ())
        self.assertEqual(protocol.features[1, 1, :2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(protocol.features[0, 0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(protocol.features[0, 0, 3].item(), 1.0, places=6)
        self.assertAlmostEqual(protocol.features[144, 0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(protocol.features[144, 0, 3].item(), -1.0, places=6)

    def test_persistence_uses_latest_observed_history(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3)
        batch = next(batches(protocol, protocol.validation, batch_size=8))
        prediction = _baseline("persistence", protocol, protocol.validation)

        self.assertEqual(prediction[0, 0].tolist(), [29.0, 29.0, 29.0])
        self.assertEqual(prediction[0, 1].tolist(), [60.0, 60.0, 60.0])
        expected_observations = int(batch.observed.sum().item())
        self.assertEqual(baselines(protocol, protocol.validation).persistence.overall.observations, expected_observations)

    def test_partial_batches_are_padded_without_new_targets(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3)
        batch = next(batches(protocol, protocol.validation, batch_size=8))
        padded = execution_batch(batch, Device.DEFAULT, batch_size=8)

        self.assertEqual(padded.values.shape, (8, 2, 4, 2))
        self.assertEqual(padded.anchor.shape, (8, 4, 1))
        self.assertEqual(int(padded.observed.sum().item()), int(batch.observed.sum().item()))

    def test_false_graph_is_an_isomorphic_relabeling(self) -> None:
        selected = graphs(dataset().graph)

        self.assertNotEqual(
            tuple(zip(selected["true"].source, selected["true"].target)),
            tuple(zip(selected["permuted"].source, selected["permuted"].target)),
        )
        self.assertEqual(selected["true"].edges, selected["permuted"].edges)
        self.assertEqual(
            sorted(selected["true"].in_degree(device=Device.DEFAULT).tolist()),
            sorted(selected["permuted"].in_degree(device=Device.DEFAULT).tolist()),
        )
        self.assertEqual(selected["self"].edges, selected["self"].nodes)

    def test_diffusion_controls_preserve_affinity_and_parameter_shape(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        selected = operators(protocol, "diffusion_gru", Device.DEFAULT)

        self.assertEqual(
            selected["true"].forward_weight.tolist(),
            selected["permuted"].forward_weight.tolist(),
        )
        self.assertEqual(selected["self"].forward_weight.tolist(), [1.0] * 4)
        self.assertEqual(selected["self"].reverse_weight.tolist(), [1.0] * 4)

    def test_residual_head_starts_at_latest_speed(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3)
        batch = next(batches(protocol, protocol.train, batch_size=17))
        model = A3Forecast(2, 2, 2, 3, head="residual")

        prediction = model(batch.values, protocol.data.graph, batch.anchor)
        expected = batch.anchor.expand(*prediction.shape)

        self.assertEqual(prediction.tolist(), expected.tolist())

    def test_diffusion_residual_starts_at_latest_speed(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        batch = next(batches(protocol, protocol.train, batch_size=17))
        diffusion = operators(protocol, "diffusion_gru", Device.DEFAULT)["true"]
        model = DiffusionForecast(6, 2, 2, 3, head="residual")

        prediction = model(batch.values, diffusion, batch.anchor)
        expected = batch.anchor.expand(*prediction.shape)

        self.assertEqual(prediction.tolist(), expected.tolist())

    def test_local_diffusion_starts_at_persistence_with_a_closed_gate(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        batch = next(batches(protocol, protocol.train, batch_size=17))
        diffusion = operators(protocol, "local_diffusion", Device.DEFAULT)["true"]
        model = LocalDiffusionForecast(6, 2, 2, 3, head="residual")

        prediction = model(batch.values, diffusion, batch.anchor)
        expected = batch.anchor.expand(*prediction.shape)

        self.assertEqual(model.spatial_gate.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(prediction.tolist(), expected.tolist())

    def test_transport_is_zero_only_for_self_diffusion(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        values = protocol.features[2]
        selected = operators(protocol, "local_diffusion", Device.DEFAULT)
        self_fields = selected["self"](values)
        true_fields = selected["true"](values)

        self.assertEqual(max((field - values).abs().max().item() for field in self_fields), 0.0)
        self.assertGreater(max((field - values).abs().max().item() for field in true_fields), 0.0)

    def test_closed_spatial_gate_opens_before_training_transport(self) -> None:
        Tensor.manual_seed(0)
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        batch = next(batches(protocol, protocol.train, batch_size=17))
        diffusion = operators(protocol, "local_diffusion", Device.DEFAULT)["true"]
        model = LocalDiffusionForecast(6, 2, 2, 3, head="residual")
        optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=0.01)

        with Context(TRAINING=1):
            optimizer.zero_grad()
            loss = (model(batch.values, diffusion, batch.anchor) - batch.target).square().mean().backward()
            assert model.spatial_gate.grad is not None
            self.assertGreater(model.spatial_gate.grad.abs().sum().item(), 0.0)
            loss.realize(*optimizer.schedule_step())
        self.assertGreater(model.spatial_gate.abs().sum().item(), 0.0)

        with Context(TRAINING=1):
            optimizer.zero_grad()
            (model(batch.values, diffusion, batch.anchor) - batch.target).square().mean().backward()
            assert model.spatial.candidate.weight.grad is not None
            self.assertGreater(model.spatial.candidate.weight.grad.abs().sum().item(), 0.0)

    def test_diffusion_forecast_does_not_materialize_product_adjacency(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        batch = next(batches(protocol, protocol.train, batch_size=2))
        diffusion = operators(protocol, "diffusion_gru", Device.DEFAULT)["true"]
        output = DiffusionForecast(6, 2, 2, 3)(batch.values, diffusion)
        shapes = {
            tuple(int(size) for size in uop._shape)
            for uop in output.uop.toposort()
            if uop._shape is not None
        }

        self.assertTrue({(4, 4), (4, 8), (8, 8)}.isdisjoint(shapes))

    def test_local_diffusion_does_not_materialize_product_adjacency(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3, feature_set="calendar")
        batch = next(batches(protocol, protocol.train, batch_size=2))
        diffusion = operators(protocol, "local_diffusion", Device.DEFAULT)["true"]
        output = LocalDiffusionForecast(6, 3, 2, 3)(batch.values, diffusion)
        shapes = {
            tuple(int(size) for size in uop._shape)
            for uop in output.uop.toposort()
            if uop._shape is not None
        }

        self.assertTrue({(4, 4), (4, 8), (8, 8)}.isdisjoint(shapes))

    def test_residual_anchor_is_causal_persistence(self) -> None:
        protocol = prepare(dataset(), history=2, horizon=3)
        batch = next(batches(protocol, protocol.validation, batch_size=8))
        expected = _baseline("persistence", protocol, protocol.validation)

        actual = protocol.standardizer.restore(batch.anchor).expand(*expected.shape)

        self.assertLess((actual - expected).abs().max().item(), 1e-5)

    def test_masked_objectives(self) -> None:
        error = Tensor([1.0, -2.0, 100.0])
        observed = Tensor([True, True, False])

        self.assertAlmostEqual(_objective(error, observed, "mse").item(), 2.5)
        self.assertAlmostEqual(_objective(error, observed, "mae").item(), 1.5)
        self.assertAlmostEqual(_objective(error, observed, "huber").item(), 1.0)

    def test_tiny_forecast_trains_through_masked_a3tgcn(self) -> None:
        result = train(
            prepare(dataset(), history=2, horizon=3),
            topologies=("true",),
            seeds=(0,),
            epochs=1,
            batch_size=17,
            hidden_features=2,
            learning_rate=0.01,
            checkpoint_every=1,
            evaluate_test=True,
        )

        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.architecture, "a3tgcn")
        self.assertEqual(result.head, "direct")
        self.assertEqual(result.loss, "mse")
        model = result.results[0]
        self.assertEqual((model.topology, model.seed, model.sparse_calls), ("true", 0, 2))
        self.assertTrue(isfinite(model.validation.overall.rmse))
        self.assertIsNotNone(model.test)
        assert model.test is not None
        self.assertTrue(isfinite(model.test.overall.rmse))
        self.assertEqual(result.protocol.pygt_windows, 26)
        self.assertEqual(result.protocol.pygt_train_windows, 20)

    def test_tiny_forecast_trains_through_sequential_diffusion(self) -> None:
        result = train(
            prepare(dataset(steps=60), history=2, horizon=3, feature_set="calendar"),
            topologies=("true",),
            seeds=(0,),
            epochs=1,
            batch_size=38,
            hidden_features=2,
            learning_rate=0.01,
            checkpoint_every=1,
            head="residual",
            loss="mae",
            architecture="diffusion_gru",
        )

        self.assertEqual(result.architecture, "diffusion_gru")
        self.assertEqual(result.protocol.feature_set, "calendar")
        self.assertEqual(result.results[0].sparse_calls, 8)
        self.assertTrue(isfinite(result.results[0].validation.overall.rmse))

    def test_tiny_forecast_trains_through_local_diffusion(self) -> None:
        result = train(
            prepare(dataset(steps=60), history=2, horizon=3, feature_set="calendar"),
            topologies=("true",),
            seeds=(0,),
            epochs=1,
            batch_size=38,
            hidden_features=2,
            learning_rate=0.01,
            checkpoint_every=1,
            head="residual",
            loss="mae",
            architecture="local_diffusion",
        )

        self.assertEqual(result.architecture, "local_diffusion")
        self.assertEqual(result.results[0].sparse_calls, 4)
        self.assertTrue(isfinite(result.results[0].validation.overall.rmse))

    def test_checkpoint_evaluation_observes_training(self) -> None:
        result = train(
            prepare(dataset(steps=60), history=2, horizon=3),
            topologies=("true",),
            seeds=(0,),
            epochs=1,
            batch_size=2,
            hidden_features=2,
            learning_rate=0.01,
            checkpoint_every=1,
        )

        before, after = result.results[0].checkpoints
        self.assertNotEqual(before.validation, after.validation)
        self.assertIsNone(result.results[0].test)

    def test_rejects_invalid_training_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            forecast(Device.DEFAULT, seeds=(True,))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            forecast(Device.DEFAULT, epochs=0)
        with self.assertRaisesRegex(ValueError, "direct.*residual"):
            forecast(Device.DEFAULT, head="other")
        with self.assertRaisesRegex(ValueError, "mse.*mae.*huber"):
            forecast(Device.DEFAULT, loss="other")
        with self.assertRaisesRegex(ValueError, "boolean"):
            forecast(Device.DEFAULT, evaluate_test=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "a3tgcn.*diffusion_gru.*local_diffusion"):
            forecast(Device.DEFAULT, architecture="other")
        with self.assertRaisesRegex(ValueError, "calendar"):
            train(
                prepare(dataset(), history=2, horizon=3),
                topologies=("true",),
                seeds=(0,),
                epochs=1,
                batch_size=17,
                hidden_features=2,
                learning_rate=0.01,
                checkpoint_every=1,
                architecture="diffusion_gru",
            )


if __name__ == "__main__":
    unittest.main()
