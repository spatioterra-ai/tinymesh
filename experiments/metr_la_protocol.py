"""Leakage-safe METR-LA windows, controls, and metrics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import cos, sin, sqrt, tau
from random import Random

from tinygrad import Tensor

from tinymesh.datasets import METRLA


@dataclass(frozen=True, eq=False)
class Standardizer:
    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, values: Tensor, observed: Tensor) -> Standardizer:
        weight = observed.cast(values.dtype)
        count = weight.sum().realize()
        if count.item() == 0:
            raise ValueError("training rows must contain an observed speed")
        mean = ((values * weight).sum() / count).realize()
        variance = (((values - mean).square() * weight).sum() / count).realize()
        return cls(mean, (variance != 0).where(variance.sqrt(), 1).realize())

    def normalize(self, values: Tensor, observed: Tensor) -> Tensor:
        return observed.where((values - self.mean) / self.scale, 0)

    def restore(self, values: Tensor) -> Tensor:
        return values * self.scale + self.mean


@dataclass(frozen=True)
class WindowSpan:
    first_target: int
    stop: int
    history: int
    horizon: int

    @property
    def windows(self) -> int:
        return self.stop - self.first_target - self.horizon + 1

    def starts(self) -> range:
        return range(self.first_target, self.stop - self.horizon + 1)


@dataclass(frozen=True, eq=False)
class Protocol:
    data: METRLA
    features: Tensor
    feature_set: str
    feature_names: tuple[str, ...]
    feature_statistics: tuple[tuple[str, float], ...]
    target: Tensor
    observed: Tensor
    standardizer: Standardizer
    node_mean: Tensor
    seasonal_mean: Tensor
    start_slot: int
    train_end: int
    validation_end: int
    train: WindowSpan
    validation: WindowSpan
    test: WindowSpan


@dataclass(frozen=True, eq=False)
class WindowBatch:
    values: Tensor
    anchor: Tensor
    target: Tensor
    observed: Tensor
    starts: tuple[int, ...]


@dataclass(frozen=True)
class Metrics:
    observations: int
    mae: float
    rmse: float


@dataclass(frozen=True)
class HorizonMetrics:
    minutes: int
    metrics: Metrics


@dataclass(frozen=True)
class Scores:
    overall: Metrics
    horizons: tuple[HorizonMetrics, ...]


@dataclass(frozen=True)
class Baselines:
    train_mean: Scores
    persistence: Scores
    seasonal_mean: Scores


@dataclass(frozen=True)
class ProtocolObservation:
    device: str
    nodes: int
    edges: int
    steps: int
    features: int
    feature_set: str
    feature_names: tuple[str, ...]
    feature_statistics: tuple[tuple[str, float], ...]
    history: int
    horizon: int
    pygt_windows: int
    pygt_train_windows: int
    train_end: int
    validation_end: int
    train_windows: int
    validation_windows: int
    test_windows: int
    train_observations: int
    speed_mean: float
    speed_scale: float
    validation: Baselines
    test: Baselines


def prepare(
    data: METRLA,
    *,
    history: int = 12,
    horizon: int = 12,
    feature_set: str = "linear_time",
) -> Protocol:
    for name, value in (("history", history), ("horizon", horizon)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if feature_set not in ("linear_time", "calendar"):
        raise ValueError("feature_set must be 'linear_time' or 'calendar'")
    steps = data.speed.shape[0]
    train_end, validation_end = int(0.7 * steps), int(0.8 * steps)
    spans = (
        WindowSpan(history, train_end, history, horizon),
        WindowSpan(train_end, validation_end, history, horizon),
        WindowSpan(validation_end, steps, history, horizon),
    )
    if any(span.windows <= 0 for span in spans):
        raise ValueError("METR-LA is too short for non-empty 70/10/20 target windows")

    observed = data.observed.realize()
    standardizer = Standardizer.fit(data.speed[:train_end], observed[:train_end])
    target = standardizer.normalize(data.speed, observed).realize()
    features, feature_names, feature_statistics = _features(
        data,
        target,
        observed,
        train_end,
        feature_set,
    )
    node_mean = _node_mean(data.speed[:train_end], observed[:train_end], standardizer.mean)
    start_slot = (data.timestamps[0].hour * 60 + data.timestamps[0].minute) // data.sample_minutes
    seasonal_mean = _seasonal_mean(data.speed[:train_end], observed[:train_end], node_mean, start_slot)
    return Protocol(
        data,
        features,
        feature_set,
        feature_names,
        feature_statistics,
        target,
        observed,
        standardizer,
        node_mean,
        seasonal_mean,
        start_slot,
        train_end,
        validation_end,
        *spans,
    )


def observe(protocol: Protocol) -> ProtocolObservation:
    steps = protocol.data.speed.shape[0]
    pygt_windows = steps - protocol.train.history - protocol.train.horizon + 1
    return ProtocolObservation(
        protocol.data.speed.device,
        protocol.data.graph.nodes,
        protocol.data.graph.edges,
        steps,
        protocol.features.shape[2],
        protocol.feature_set,
        protocol.feature_names,
        protocol.feature_statistics,
        protocol.train.history,
        protocol.train.horizon,
        pygt_windows,
        int(0.8 * pygt_windows),
        protocol.train_end,
        protocol.validation_end,
        protocol.train.windows,
        protocol.validation.windows,
        protocol.test.windows,
        int(protocol.observed[:protocol.train_end].sum().item()),
        float(protocol.standardizer.mean.item()),
        float(protocol.standardizer.scale.item()),
        baselines(protocol, protocol.validation),
        baselines(protocol, protocol.test),
    )


def _features(
    data: METRLA,
    target: Tensor,
    observed: Tensor,
    train_end: int,
    feature_set: str,
) -> tuple[Tensor, tuple[str, ...], tuple[tuple[str, float], ...]]:
    steps, nodes = data.speed.shape
    day = [
        (timestamp.hour * 60 + timestamp.minute) / (24 * 60)
        for timestamp in data.timestamps
    ]
    if feature_set == "linear_time":
        time = Tensor(day, dtype=data.speed.dtype, device=data.speed.device).realize()
        mean = time[:train_end].mean().realize()
        scale = time[:train_end].std(correction=0).realize()
        if scale.item() == 0:
            raise ValueError("training timestamps must vary within the day")
        time = ((time - mean) / scale).reshape(steps, 1, 1).expand(steps, nodes, 1)
        return (
            target.unsqueeze(2).cat(time, dim=2).realize(),
            ("speed", "time_of_day"),
            (
                ("time_mean", float(mean.item())),
                ("time_scale", float(scale.item())),
            ),
        )

    calendar = Tensor(
        [
            (
                sin(tau * phase),
                cos(tau * phase),
                sin(tau * (timestamp.weekday() + phase) / 7),
                cos(tau * (timestamp.weekday() + phase) / 7),
            )
            for timestamp, phase in zip(data.timestamps, day)
        ],
        dtype=data.speed.dtype,
        device=data.speed.device,
    ).reshape(steps, 1, 4).expand(steps, nodes, 4)
    features = target.unsqueeze(2).cat(
        observed.cast(data.speed.dtype).unsqueeze(2),
        calendar,
        dim=2,
    )
    return (
        features.realize(),
        ("speed", "observed", "sin_day", "cos_day", "sin_week", "cos_week"),
        (),
    )


def batches(
    protocol: Protocol,
    span: WindowSpan,
    batch_size: int,
    *,
    shuffle: int | None = None,
    tensors: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Iterator[WindowBatch]:
    starts = list(span.starts())
    if shuffle is not None:
        Random(shuffle).shuffle(starts)
    features, target, observed = (
        (protocol.features, protocol.target, protocol.observed)
        if tensors is None
        else tensors
    )
    fallback = ((protocol.node_mean - protocol.standardizer.mean) / protocol.standardizer.scale).to(features.device).realize()
    for offset in range(0, len(starts), batch_size):
        selected = tuple(starts[offset:offset + batch_size])
        start = Tensor(selected, device=features.device).reshape(-1, 1)
        history_index = start + Tensor(tuple(range(-span.history, 0)), device=features.device).reshape(1, -1)
        target_index = start + Tensor(tuple(range(span.horizon)), device=features.device).reshape(1, -1)
        values, history_observed = features[history_index], observed[history_index]
        anchor = fallback.reshape(1, protocol.data.graph.nodes).expand(len(selected), protocol.data.graph.nodes)
        for period in range(span.history):
            anchor = history_observed[:, period].where(values[:, period, :, 0], anchor)
        yield WindowBatch(
            values,
            anchor.unsqueeze(2),
            target[target_index].permute(0, 2, 1),
            observed[target_index].permute(0, 2, 1),
            selected,
        )


def baselines(protocol: Protocol, span: WindowSpan) -> Baselines:
    target, observed = _targets(protocol, span)
    return Baselines(
        *(score(((_baseline(kind, protocol, span) - target, observed),)) for kind in ("train_mean", "persistence", "seasonal"))
    )


def score(errors: Iterable[tuple[Tensor, Tensor]]) -> Scores:
    absolute: list[float] | None = None
    square: list[float] | None = None
    count: list[int] | None = None
    for error, observed in errors:
        weight = observed.cast(error.dtype)
        terms = (
            (error.abs() * weight).sum(axis=(0, 1)),
            (error.square() * weight).sum(axis=(0, 1)),
            weight.sum(axis=(0, 1)),
        )
        Tensor.realize(*terms)
        if absolute is None:
            absolute = [0.0] * terms[0].shape[0]
            square = [0.0] * terms[1].shape[0]
            count = [0] * terms[2].shape[0]
        for horizon, values in enumerate(zip(*(term.tolist() for term in terms))):
            absolute[horizon] += values[0]
            square[horizon] += values[1]
            count[horizon] += int(values[2])
    if absolute is None or square is None or count is None or any(value == 0 for value in count):
        raise ValueError("every forecast horizon must contain an observed target")

    observations = sum(count)
    overall = Metrics(observations, sum(absolute) / observations, sqrt(sum(square) / observations))
    horizons = tuple(
        HorizonMetrics(
            (index + 1) * 5,
            Metrics(count[index], absolute[index] / count[index], sqrt(square[index] / count[index])),
        )
        for index in (2, 5, 11)
        if index < len(count)
    )
    return Scores(overall, horizons)


def _node_mean(values: Tensor, observed: Tensor, fallback: Tensor) -> Tensor:
    weight = observed.cast(values.dtype)
    count = weight.sum(axis=0)
    mean = (values * weight).sum(axis=0) / count.maximum(1)
    return (count != 0).where(mean, fallback).realize()


def _seasonal_mean(values: Tensor, observed: Tensor, fallback: Tensor, start_slot: int) -> Tensor:
    slots = 24 * 60 // 5
    means = []
    for slot in range(slots):
        first = (slot - start_slot) % slots
        means.append(
            fallback
            if first >= values.shape[0]
            else _node_mean(values[first::slots], observed[first::slots], fallback)
        )
    return Tensor.stack(*means).realize()


def _targets(protocol: Protocol, span: WindowSpan) -> tuple[Tensor, Tensor]:
    slices = tuple(
        slice(span.first_target + offset, span.first_target + offset + span.windows)
        for offset in range(span.horizon)
    )
    return (
        Tensor.stack(*(protocol.data.speed[index] for index in slices), dim=2),
        Tensor.stack(*(protocol.observed[index] for index in slices), dim=2),
    )


def _baseline(kind: str, protocol: Protocol, span: WindowSpan) -> Tensor:
    nodes, windows = protocol.data.graph.nodes, span.windows
    if kind == "train_mean":
        return protocol.node_mean.reshape(1, nodes, 1).expand(windows, nodes, span.horizon)
    if kind == "seasonal":
        slots = protocol.seasonal_mean.shape[0]
        row = Tensor.arange(windows).to(protocol.data.speed.device) + protocol.start_slot + span.first_target
        return Tensor.stack(
            *(protocol.seasonal_mean[(row + offset) % slots] for offset in range(span.horizon)),
            dim=2,
        )
    if kind != "persistence":
        raise ValueError(kind)

    latest = protocol.node_mean.reshape(1, nodes).expand(windows, nodes)
    for offset in range(span.history):
        start = span.first_target - span.history + offset
        latest = protocol.observed[start:start + windows].where(
            protocol.data.speed[start:start + windows],
            latest,
        )
    return latest.unsqueeze(2).expand(windows, nodes, span.horizon)
