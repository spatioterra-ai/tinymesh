import unittest
from math import sqrt

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from experiments.montevideo_delayed_edges import (
    _compare,
    _fit,
    _incoming_mean,
    _predict,
    _score,
)
from tinymesh import Graph, StaticGraphTemporalSignal


SOURCE = (0, 2, 1)
TARGET = (1, 1, 2)


def host_mean(field: list[list[float]]) -> list[list[float]]:
    output = [[0.0] * 4 for _ in field]
    degree = [TARGET.count(node) for node in range(4)]
    for time, values in enumerate(field):
        for source, target in zip(SOURCE, TARGET):
            output[time][target] += values[source]
        output[time] = [
            value / degree[node] if degree[node] else 0.0
            for node, value in enumerate(output[time])
        ]
    return output


def host_metrics(
    prediction: list[list[float]],
    target: list[list[float]],
) -> tuple[float, float]:
    error = [
        predicted - observed
        for predicted_row, target_row in zip(prediction, target)
        for predicted, observed in zip(predicted_row, target_row)
    ]
    return (
        sum(abs(value) for value in error) / len(error),
        sqrt(sum(value * value for value in error) / len(error)),
    )


def tensor(values: list[list[float]]) -> Tensor:
    return Tensor(values, device=Device.DEFAULT).reshape(len(values), 4, 1).realize()


def periodic_signal(raw: list[list[float]]) -> StaticGraphTemporalSignal:
    return StaticGraphTemporalSignal(
        Graph(4, [0, 1, 2], [1, 2, 0]),
        ("a", "b", "c", "isolated"),
        tensor(raw[:-1]),
        tensor(raw[1:]),
    )


class MontevideoDelayedEdgesTest(unittest.TestCase):
    def test_matches_independent_directed_periodic_reference(self) -> None:
        baseline = [
            [float((time % 3) * (node + 1)) for node in range(4)]
            for time in range(10)
        ]
        residual = [
            [float(((time + 2 * node) % 5) - 2) for node in range(4)]
            for time in range(10)
        ]
        target = [
            [base + change for base, change in zip(base_row, residual_row)]
            for base_row, residual_row in zip(baseline, residual)
        ]
        aggregate = host_mean(residual)
        lag, train_end = 2, 7
        numerator = sum(
            aggregate[row - lag][node] * residual[row][node]
            for row in range(lag, train_end)
            for node in range(4)
        )
        denominator = sum(
            aggregate[row - lag][node] ** 2
            for row in range(lag, train_end)
            for node in range(4)
        )
        alpha = numerator / denominator
        prediction = [
            [
                baseline[row][node] + alpha * aggregate[row - lag][node]
                for node in range(4)
            ]
            for row in range(train_end, 10)
        ]
        mae, rmse = host_metrics(prediction, target[train_end:])

        graph = Graph(4, SOURCE, TARGET)
        actual_aggregate = _incoming_mean(graph, tensor(residual))
        actual_alpha = _fit(actual_aggregate, tensor(residual), lag, train_end)
        actual_prediction = _predict(
            tensor(baseline),
            actual_aggregate,
            lag,
            actual_alpha,
            train_end,
            10,
        )
        metrics = _score(actual_prediction, tensor(target)[train_end:])

        self.assertEqual(actual_aggregate.tolist(), tensor(aggregate).tolist())
        self.assertAlmostEqual(actual_alpha, alpha, places=6)
        self.assertAlmostEqual(metrics.mae, mae, places=6)
        self.assertAlmostEqual(metrics.rmse, rmse, places=6)

    def test_lag_reads_only_rows_before_each_target(self) -> None:
        baseline = tensor([[float(time)] * 4 for time in range(10)])
        aggregate = tensor([[float(time + node) for node in range(4)] for time in range(10)])
        prediction = _predict(baseline, aggregate, 2, 7, 7, 10)
        changed = aggregate[:8].cat(aggregate[8:] + 1000, dim=0)

        self.assertEqual(
            prediction.tolist(),
            [
                [[42.0], [49.0], [56.0], [63.0]],
                [[50.0], [57.0], [64.0], [71.0]],
                [[58.0], [65.0], [72.0], [79.0]],
            ],
        )
        self.assertEqual(
            _predict(baseline, changed, 2, 7, 7, 10).tolist(),
            prediction.tolist(),
        )

    def test_test_rows_do_not_change_fit_or_selection(self) -> None:
        raw = [
            [float((time % 168) % (node + 2)) for node in range(4)]
            for time in range(221)
        ]
        changed_raw = [
            [value + (1000.0 if time >= 201 else 0.0) for value in row]
            for time, row in enumerate(raw)
        ]
        original = _compare(periodic_signal(raw), train_end=180, validation_end=200)
        changed = _compare(
            periodic_signal(changed_raw),
            train_end=180,
            validation_end=200,
        )

        self.assertEqual(changed.floor.validation, original.floor.validation)
        self.assertEqual(
            [
                (structure.candidates, structure.selected, structure.validation_blocks)
                for structure in changed.structures
            ],
            [
                (structure.candidates, structure.selected, structure.validation_blocks)
                for structure in original.structures
            ],
        )
        self.assertNotEqual(changed.floor.test, original.floor.test)
        self.assertEqual(
            [structure.name for structure in original.structures],
            ["real", "reverse", "permuted"],
        )
        self.assertEqual(
            [structure.covered_nodes for structure in original.structures],
            [3, 3, 3],
        )
        self.assertEqual(
            [structure.sparse_calls for structure in original.structures],
            [1, 1, 1],
        )
        self.assertEqual(original.sum_topology_int32, 2 * (4 + 1 + 3))
        self.assertFalse(original.gate.validation_passed)
        self.assertFalse(original.gate.test_passed)
        self.assertFalse(original.gate.confirmed)

    def test_zero_signal_fits_zero_alpha(self) -> None:
        values = Tensor.zeros(30, 4, 1, device=Device.DEFAULT).realize()

        self.assertEqual(_fit(values, values, lag=24, train_end=25), 0.0)

    def test_mean_uses_one_sparse_call_without_dense_topology(self) -> None:
        graph = Graph(4, SOURCE, TARGET)
        field = tensor([[float(time + node) for node in range(4)] for time in range(8)])
        aggregate = _incoming_mean(graph, field)
        calls = [uop for uop in aggregate.uop.toposort() if uop.op is Ops.CALL]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].src[0].arg.name, "csr_sum")
        self.assertEqual(
            [source._shape for source in calls[0].src[2:]],
            [(4, 8), (5,), (3,), (5,), (3,)],
        )
        self.assertNotIn(
            (graph.nodes, graph.nodes),
            [uop._shape for uop in aggregate.uop.toposort()],
        )

    def test_rejects_short_splits(self) -> None:
        raw = [[0.0] * 4 for _ in range(221)]
        with self.assertRaisesRegex(ValueError, "training lag history"):
            _compare(periodic_signal(raw), train_end=24, validation_end=200)
        with self.assertRaisesRegex(ValueError, "at least three"):
            _compare(periodic_signal(raw), train_end=180, validation_end=219)


if __name__ == "__main__":
    unittest.main()
