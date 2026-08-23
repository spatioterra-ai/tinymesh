import unittest
from math import sqrt

from tinygrad import Device, Tensor

from experiments.directed_gru import DiffusionForecast
from experiments.montevideo_forecast import (
    LocalForecast,
    Standardizer,
    _baselines,
    _model,
    forecast,
    normalize_splits,
    split_signal,
)
from tinymesh import Graph, StaticGraphTemporalSignal
from tinymesh.nn import DirectedDiffusion


def signal() -> StaticGraphTemporalSignal:
    return StaticGraphTemporalSignal(
        Graph(2, [0, 1], [1, 0]),
        ("left", "right"),
        Tensor(
            [[[float(time)], [float(2 * time)]] for time in range(10)],
            device=Device.DEFAULT,
        ).realize(),
        Tensor(
            [[[float(time + 1)], [float(2 * time + 2)]] for time in range(10)],
            device=Device.DEFAULT,
        ).realize(),
    )


class MontevideoForecastTest(unittest.TestCase):
    def assert_metrics(self, actual, *, mae, rmse) -> None:
        self.assertAlmostEqual(actual.mae, mae, places=5)
        self.assertAlmostEqual(actual.rmse, rmse, places=5)

    def test_target_time_split_retains_boundary_history(self) -> None:
        dataset = signal()
        splits = split_signal(dataset, history=2)

        self.assertEqual((splits.train_end, splits.validation_end), (7, 8))
        self.assertEqual(
            (len(splits.train), len(splits.validation), len(splits.test)),
            (7, 2, 3),
        )
        self.assertEqual(splits.validation.x.tolist(), dataset.x[6:8].tolist())
        self.assertEqual(splits.test.x.tolist(), dataset.x[7:].tolist())
        self.assertEqual(
            [target.tolist() for _, target in splits.validation.batches(batch_size=4, history=2)],
            [[[[8.0], [16.0]]]],
        )
        self.assertEqual(
            [target.tolist() for _, target in splits.test.batches(batch_size=4, history=2)],
            [[[[9.0], [18.0]], [[10.0], [20.0]]]],
        )

    def test_normalization_uses_training_rows_only(self) -> None:
        dataset = signal()
        splits = split_signal(dataset, history=2)
        _, input_standardizer, target_standardizer = normalize_splits(splits)
        changed = StaticGraphTemporalSignal(
            dataset.graph,
            dataset.node_ids,
            dataset.x[:7].cat(dataset.x[7:] + 1000, dim=0),
            dataset.y[:7].cat(dataset.y[7:] - 1000, dim=0),
        )
        _, changed_input, changed_target = normalize_splits(
            split_signal(changed, history=2)
        )

        self.assertEqual(changed_input.mean.tolist(), input_standardizer.mean.tolist())
        self.assertEqual(changed_input.scale.tolist(), input_standardizer.scale.tolist())
        self.assertEqual(changed_target.mean.tolist(), target_standardizer.mean.tolist())
        self.assertEqual(changed_target.scale.tolist(), target_standardizer.scale.tolist())

    def test_standardizer_restores_raw_units_and_constant_nodes(self) -> None:
        values = Tensor(
            [[[2.0], [7.0]], [[4.0], [7.0]], [[6.0], [7.0]]],
            device=Device.DEFAULT,
        ).realize()
        standardizer = Standardizer.fit(values)

        self.assertEqual(standardizer.mean.tolist(), [[4.0], [7.0]])
        self.assertAlmostEqual(standardizer.scale[0].item(), sqrt(8 / 3), places=5)
        self.assertEqual(standardizer.scale[1].item(), 1.0)
        restored = standardizer.restore(standardizer.normalize(values)).realize()
        self.assertEqual(restored.tolist(), values.tolist())

    def test_raw_unit_baselines_match_reference(self) -> None:
        splits = split_signal(signal(), history=2)
        train_mean = splits.train.y.mean(axis=0).realize()
        validation = _baselines(
            splits.validation,
            train_mean,
            batch_size=4,
            history=2,
        )
        test = _baselines(splits.test, train_mean, batch_size=4, history=2)

        self.assertEqual((validation.targets, validation.zero_fraction), (2, 0.0))
        self.assert_metrics(validation.zero, mae=12.0, rmse=sqrt(160))
        self.assert_metrics(validation.persistence, mae=1.5, rmse=sqrt(2.5))
        self.assert_metrics(validation.train_mean, mae=6.0, rmse=sqrt(40))
        self.assertEqual((test.targets, test.zero_fraction), (4, 0.0))
        self.assert_metrics(test.zero, mae=14.25, rmse=sqrt(226.25))
        self.assert_metrics(test.persistence, mae=1.5, rmse=sqrt(2.5))
        self.assert_metrics(test.train_mean, mae=8.25, rmse=sqrt(76.25))

    def test_models_bind_prediction_dependencies_once(self) -> None:
        graph = Graph(2, [0, 1], [1, 0])
        diffusion = {
            "unit": DirectedDiffusion(
                graph,
                Tensor.ones(graph.edges, device=Device.DEFAULT).realize(),
            )
        }
        values = Tensor.zeros(2, 3, graph.nodes, 1, device=Device.DEFAULT)

        for name, expected in (("lstm", LocalForecast), ("unit", DiffusionForecast)):
            with self.subTest(model=name):
                forecast_arm = _model(name, hidden_features=2, diffusion=diffusion)
                self.assertIsInstance(forecast_arm.model, expected)
                self.assertEqual(forecast_arm(values, realize_steps=True).shape, (2, graph.nodes, 1))

    def test_rejects_invalid_or_short_split(self) -> None:
        for history in (0, True):
            with self.subTest(history=history), self.assertRaisesRegex(
                ValueError,
                "positive integer",
            ):
                split_signal(signal(), history=history)
        with self.assertRaisesRegex(ValueError, "too short"):
            split_signal(signal(), history=8)

    def test_rejects_invalid_training_configuration_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            forecast(Device.DEFAULT, seeds=(True,))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            forecast(Device.DEFAULT, epochs=0)


if __name__ == "__main__":
    unittest.main()
