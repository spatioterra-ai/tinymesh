import unittest
from datetime import datetime, timedelta
from math import isfinite

from tinygrad import Device, Tensor

from experiments.metr_la_forecast import (
    _execution_batch,
    _graphs,
    forecast,
    train,
)
from experiments.metr_la_protocol import _baseline, baselines, batches, prepare
from tinymesh import Graph
from tinymesh.datasets import METRLA


def dataset(*, future_shift: float = 0.0) -> METRLA:
    graph = Graph(
        4,
        [0, 1, 2, 3, 0, 1, 2, 2],
        [0, 1, 2, 3, 1, 2, 0, 3],
    )
    rows = []
    for time in range(30):
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
        tuple(datetime(2012, 3, 1) + timedelta(minutes=5 * time) for time in range(30)),
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
        padded = _execution_batch(batch, Device.DEFAULT, batch_size=8)

        self.assertEqual(padded.values.shape, (8, 2, 4, 2))
        self.assertEqual(int(padded.observed.sum().item()), int(batch.observed.sum().item()))

    def test_false_graph_is_an_isomorphic_relabeling(self) -> None:
        graphs = _graphs(dataset().graph)

        self.assertNotEqual(
            tuple(zip(graphs["true"].source, graphs["true"].target)),
            tuple(zip(graphs["permuted"].source, graphs["permuted"].target)),
        )
        self.assertEqual(graphs["true"].edges, graphs["permuted"].edges)
        self.assertEqual(
            sorted(graphs["true"].in_degree(device=Device.DEFAULT).tolist()),
            sorted(graphs["permuted"].in_degree(device=Device.DEFAULT).tolist()),
        )
        self.assertEqual(graphs["self"].edges, graphs["self"].nodes)

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
        )

        self.assertEqual(len(result.results), 1)
        model = result.results[0]
        self.assertEqual((model.topology, model.seed, model.sparse_calls), ("true", 0, 2))
        self.assertTrue(isfinite(model.validation.overall.rmse))
        self.assertTrue(isfinite(model.test.overall.rmse))
        self.assertEqual(result.protocol.pygt_windows, 26)
        self.assertEqual(result.protocol.pygt_train_windows, 20)

    def test_rejects_invalid_training_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            forecast(Device.DEFAULT, seeds=(True,))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            forecast(Device.DEFAULT, epochs=0)


if __name__ == "__main__":
    unittest.main()
