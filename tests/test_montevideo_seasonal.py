import unittest
from math import sqrt

from tinygrad import Device, Tensor

from experiments.montevideo_seasonal import _compare, _phase_mean
from tinymesh import Graph, StaticGraphTemporalSignal


PERIOD = 168
BASELINES = ("zero", "persistence", "daily", "weekly", "hour_of_week")


def signal(steps: int = 220) -> tuple[StaticGraphTemporalSignal, list[list[float]]]:
    raw = [
        [
            0.0 if time % PERIOD % 5 else float(time % PERIOD),
            float((time % PERIOD) % 7),
        ]
        for time in range(steps + 1)
    ]
    return (
        StaticGraphTemporalSignal(
            Graph(2, [0, 1], [1, 0]),
            ("left", "right"),
            Tensor(raw[:-1], device=Device.DEFAULT).reshape(steps, 2, 1).realize(),
            Tensor(raw[1:], device=Device.DEFAULT).reshape(steps, 2, 1).realize(),
        ),
        raw,
    )


def reference(
    raw: list[list[float]],
    name: str,
    start: int,
    end: int,
    train_end: int,
) -> tuple[int, float, float, float]:
    phase_mean = {
        phase: [
            sum(raw[row + 1][node] for row in range(train_end) if (row + 1) % PERIOD == phase)
            / sum((row + 1) % PERIOD == phase for row in range(train_end))
            for node in range(2)
        ]
        for phase in range(PERIOD)
    }
    if name == "daily":
        start = max(start, 23)
    elif name == "weekly":
        start = max(start, 167)
    errors = []
    zeros = 0
    for row in range(start, end):
        for node in range(2):
            target = raw[row + 1][node]
            if name == "zero":
                prediction = 0.0
            elif name == "persistence":
                prediction = raw[row][node]
            elif name == "daily":
                prediction = raw[row + 1 - 24][node]
            elif name == "weekly":
                prediction = raw[row + 1 - 168][node]
            else:
                prediction = phase_mean[(row + 1) % PERIOD][node]
            errors.append(prediction - target)
            zeros += target == 0
    return (
        len(errors),
        zeros / len(errors),
        sum(abs(error) for error in errors) / len(errors),
        sqrt(sum(error * error for error in errors) / len(errors)),
    )


class MontevideoSeasonalTest(unittest.TestCase):
    def test_matches_independent_periodic_reference(self) -> None:
        dataset, raw = signal()
        observation = _compare(dataset, train_end=180, validation_end=200)

        self.assertEqual(tuple(result.name for result in observation.validation), BASELINES)
        for result in observation.validation:
            targets, zero_fraction, mae, rmse = reference(
                raw,
                result.name,
                180,
                200,
                180,
            )
            self.assertEqual((result.rows, result.coverage, result.targets), (20, 1.0, targets))
            self.assertAlmostEqual(result.zero_fraction, zero_fraction, places=6)
            self.assertAlmostEqual(result.metrics.mae, mae, places=5)
            self.assertAlmostEqual(result.metrics.rmse, rmse, places=5)
        self.assertEqual(observation.selected, "weekly")
        self.assertEqual(observation.test.metrics.mae, 0.0)
        self.assertEqual(observation.test.metrics.rmse, 0.0)

    def test_daily_history_crosses_the_validation_boundary_causally(self) -> None:
        dataset, raw = signal()
        observation = _compare(dataset, train_end=180, validation_end=200)
        daily = observation.validation[2]
        expected = reference(raw, "daily", 180, 200, 180)

        self.assertEqual(daily.targets, expected[0])
        self.assertAlmostEqual(daily.metrics.rmse, expected[3], places=5)

    def test_future_rows_do_not_fit_phase_means_or_select_the_floor(self) -> None:
        dataset, _ = signal()
        original_mean, _ = _phase_mean(dataset, 180)
        changed = StaticGraphTemporalSignal(
            dataset.graph,
            dataset.node_ids,
            dataset.x[:200].cat(dataset.x[200:] + 1000, dim=0),
            dataset.y[:200].cat(dataset.y[200:] + 1000, dim=0),
        )
        changed_mean, _ = _phase_mean(changed, 180)
        original = _compare(dataset, train_end=180, validation_end=200)
        future = _compare(changed, train_end=180, validation_end=200)

        self.assertEqual(
            [mean.tolist() for mean in changed_mean],
            [mean.tolist() for mean in original_mean],
        )
        self.assertEqual(future.validation, original.validation)
        self.assertEqual(future.selected, original.selected)
        self.assertNotEqual(future.test.metrics, original.test.metrics)

    def test_unobserved_phases_use_the_training_mean(self) -> None:
        dataset, _ = signal(20)
        means, unobserved = _phase_mean(dataset, 10)
        fallback = dataset.y[:10].mean(axis=0).tolist()

        self.assertEqual(unobserved, PERIOD - 10)
        self.assertEqual(means[100].tolist(), fallback)

    def test_rejects_empty_or_unordered_splits(self) -> None:
        dataset, _ = signal()
        for boundaries in ((0, 200), (180, 180), (200, len(dataset))):
            with self.subTest(boundaries=boundaries), self.assertRaisesRegex(
                ValueError,
                "non-empty ordered",
            ):
                _compare(
                    dataset,
                    train_end=boundaries[0],
                    validation_end=boundaries[1],
                )


if __name__ == "__main__":
    unittest.main()
