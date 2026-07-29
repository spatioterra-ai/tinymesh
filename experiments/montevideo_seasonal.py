"""Select a causal seasonal floor for the Montevideo signal."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

from tinygrad import Device, Tensor

from experiments.montevideo_forecast import Metrics
from tinymesh import StaticGraphTemporalSignal
from tinymesh.datasets import montevideo_bus


DAY = 24
PERIOD = 168
BASELINES = ("zero", "persistence", "daily", "weekly", "hour_of_week")


@dataclass(frozen=True)
class Evaluation:
    name: str
    rows: int
    coverage: float
    targets: int
    zero_fraction: float
    metrics: Metrics


@dataclass(frozen=True)
class Observation:
    device: str
    nodes: int
    steps: int
    train_end: int
    validation_end: int
    period: int
    unobserved_phases: int
    validation: tuple[Evaluation, ...]
    selected: str
    test: Evaluation


def compare(device: str = Device.DEFAULT) -> Observation:
    return _compare(
        montevideo_bus(lags=1, device=device).signal,
        train_end=520,
        validation_end=594,
    )


def _compare(
    signal: StaticGraphTemporalSignal,
    *,
    train_end: int,
    validation_end: int,
) -> Observation:
    if not 0 < train_end < validation_end < len(signal):
        raise ValueError("expected non-empty ordered train, validation, and test rows")
    if validation_end < PERIOD:
        raise ValueError("validation has no target with weekly history")

    phase_mean, unobserved = _phase_mean(signal, train_end)
    validation = tuple(
        _evaluate(signal, name, train_end, validation_end, phase_mean)
        for name in BASELINES
    )
    selected = min(
        enumerate(validation),
        key=lambda item: (item[1].metrics.rmse, item[1].metrics.mae, item[0]),
    )[1].name
    return Observation(
        signal.x.device,
        signal.graph.nodes,
        len(signal),
        train_end,
        validation_end,
        PERIOD,
        unobserved,
        validation,
        selected,
        _evaluate(signal, selected, validation_end, len(signal), phase_mean),
    )


def _phase_mean(
    signal: StaticGraphTemporalSignal,
    train_end: int,
) -> tuple[tuple[Tensor, ...], int]:
    fallback = signal.y[:train_end].mean(axis=0).realize()
    means = []
    unobserved = 0
    for phase in range(PERIOD):
        first = (phase - 1) % PERIOD
        if first >= train_end:
            means.append(fallback)
            unobserved += 1
        else:
            means.append(signal.y[first:train_end:PERIOD].mean(axis=0).realize())
    return tuple(means), unobserved


def _evaluate(
    signal: StaticGraphTemporalSignal,
    name: str,
    start: int,
    end: int,
    phase_mean: tuple[Tensor, ...],
) -> Evaluation:
    requested = end - start
    if name == "daily":
        start = max(start, DAY - 1)
        prediction = signal.x[start + 1 - DAY:end + 1 - DAY]
    elif name == "weekly":
        start = max(start, PERIOD - 1)
        prediction = signal.x[start + 1 - PERIOD:end + 1 - PERIOD]
    else:
        target = signal.y[start:end]
        if name == "zero":
            prediction = target * 0
        elif name == "persistence":
            prediction = signal.x[start:end]
        elif name == "hour_of_week":
            prediction = Tensor.stack(
                *(phase_mean[(row + 1) % PERIOD] for row in range(start, end))
            )
        else:
            raise ValueError(f"unknown baseline {name!r}")

    target = signal.y[start:end]
    error = prediction - target
    absolute, square, zeros = error.abs().sum(), error.square().sum(), (target == 0).sum()
    Tensor.realize(absolute, square, zeros)
    targets = target.numel()
    return Evaluation(
        name,
        end - start,
        (end - start) / requested,
        targets,
        int(zeros.item()) / targets,
        Metrics(absolute.item() / targets, sqrt(square.item() / targets)),
    )


def main() -> None:
    print(json.dumps(asdict(compare(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
