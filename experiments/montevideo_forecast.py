"""Define the causal Montevideo forecast protocol and raw-unit baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

from tinygrad import Device, Tensor
from tinygrad.helpers import getenv

from tinymesh import StaticGraphTemporalSignal
from tinymesh.datasets import montevideo_bus


@dataclass(frozen=True, eq=False)
class Standardizer:
    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, values: Tensor) -> Standardizer:
        mean = values.mean(axis=0).realize()
        scale = values.std(axis=0, correction=0).realize()
        return cls(mean, (scale != 0).where(scale, 1).realize())

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.scale

    def restore(self, values: Tensor) -> Tensor:
        return values * self.scale + self.mean


@dataclass(frozen=True, eq=False)
class SignalSplits:
    train: StaticGraphTemporalSignal
    validation: StaticGraphTemporalSignal
    test: StaticGraphTemporalSignal
    train_end: int
    validation_end: int


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


@dataclass(frozen=True)
class Baselines:
    targets: int
    zero_fraction: float
    zero: Metrics
    persistence: Metrics
    train_mean: Metrics


@dataclass(frozen=True)
class Observation:
    device: str
    lags: int
    history: int
    batch_size: int
    nodes: int
    edges: int
    steps: int
    train_end: int
    validation_end: int
    train_windows: int
    validation_windows: int
    test_windows: int
    constant_input_nodes: int
    constant_target_nodes: int
    train: Baselines
    validation: Baselines
    test: Baselines


def split_signal(
    signal: StaticGraphTemporalSignal,
    *,
    history: int,
) -> SignalSplits:
    if not isinstance(history, int) or isinstance(history, bool) or history <= 0:
        raise ValueError("history must be a positive integer")
    train_end = int(0.7 * len(signal))
    validation_end = int(0.8 * len(signal))
    if train_end < history or validation_end == train_end or validation_end == len(signal):
        raise ValueError("signal is too short for non-empty 70/10/20 windows")
    context = history - 1
    return SignalSplits(
        signal[:train_end],
        signal[train_end - context:validation_end],
        signal[validation_end - context:],
        train_end,
        validation_end,
    )


def normalize_splits(
    splits: SignalSplits,
) -> tuple[SignalSplits, Standardizer, Standardizer]:
    input_standardizer = Standardizer.fit(splits.train.x)
    target_standardizer = Standardizer.fit(splits.train.y)
    return (
        SignalSplits(
            _normalize(splits.train, input_standardizer, target_standardizer),
            _normalize(splits.validation, input_standardizer, target_standardizer),
            _normalize(splits.test, input_standardizer, target_standardizer),
            splits.train_end,
            splits.validation_end,
        ),
        input_standardizer,
        target_standardizer,
    )


def compare(
    device: str,
    *,
    history: int = 24,
    batch_size: int = 32,
) -> Observation:
    signal = montevideo_bus(lags=1, device=device).signal
    splits = split_signal(signal, history=history)
    _, input_standardizer, target_standardizer = normalize_splits(splits)
    train_mean = splits.train.y.mean(axis=0).realize()
    settings = {"batch_size": batch_size, "history": history}
    return Observation(
        device,
        lags=1,
        history=history,
        batch_size=batch_size,
        nodes=signal.graph.nodes,
        edges=signal.graph.edges,
        steps=len(signal),
        train_end=splits.train_end,
        validation_end=splits.validation_end,
        train_windows=_windows(splits.train, history),
        validation_windows=_windows(splits.validation, history),
        test_windows=_windows(splits.test, history),
        constant_input_nodes=int(
            (splits.train.x.std(axis=0, correction=0) == 0).sum().item()
        ),
        constant_target_nodes=int(
            (splits.train.y.std(axis=0, correction=0) == 0).sum().item()
        ),
        train=_baselines(splits.train, train_mean, **settings),
        validation=_baselines(splits.validation, train_mean, **settings),
        test=_baselines(splits.test, train_mean, **settings),
    )


def _normalize(
    signal: StaticGraphTemporalSignal,
    input_standardizer: Standardizer,
    target_standardizer: Standardizer,
) -> StaticGraphTemporalSignal:
    return StaticGraphTemporalSignal(
        signal.graph,
        signal.node_ids,
        input_standardizer.normalize(signal.x).realize(),
        target_standardizer.normalize(signal.y).realize(),
        signal.edge_weight,
    )


def _baselines(
    signal: StaticGraphTemporalSignal,
    train_mean: Tensor,
    *,
    batch_size: int,
    history: int,
) -> Baselines:
    absolute = [0.0, 0.0, 0.0]
    square = [0.0, 0.0, 0.0]
    targets = zeros = 0
    for values, target in signal.batches(batch_size=batch_size, history=history):
        predictions = (target * 0, values[:, -1], train_mean.expand(*target.shape))
        errors = [prediction - target for prediction in predictions]
        terms = [term for error in errors for term in (error.abs().sum(), error.square().sum())]
        zero_count = (target == 0).sum()
        Tensor.realize(*terms, zero_count)
        for index in range(3):
            absolute[index] += terms[2 * index].item()
            square[index] += terms[2 * index + 1].item()
        targets += target.numel()
        zeros += int(zero_count.item())
    metrics = tuple(
        Metrics(error / targets, sqrt(squared_error / targets))
        for error, squared_error in zip(absolute, square)
    )
    return Baselines(targets, zeros / targets, *metrics)


def _windows(signal: StaticGraphTemporalSignal, history: int) -> int:
    return len(signal) - history + 1


def main() -> None:
    observation = compare(
        Device.DEFAULT,
        history=getenv("HISTORY", 24),
        batch_size=getenv("BS", 32),
    )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
