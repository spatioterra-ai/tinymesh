"""Test delayed neighbor residuals beyond the Montevideo seasonal floor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

from tinygrad import Device, Tensor
from tinygrad.uop.ops import Ops

from experiments.montevideo_forecast import Metrics
from experiments.montevideo_seasonal import PERIOD, _phase_mean
from tinymesh import Graph, StaticGraphTemporalSignal
from tinymesh.datasets import montevideo_bus


LAGS = (1, 2, 3, 6, 12, 24)


@dataclass(frozen=True)
class Candidate:
    lag: int
    alpha: float
    validation: Metrics


@dataclass(frozen=True)
class Structure:
    name: str
    covered_nodes: int
    node_coverage: float
    sparse_calls: int
    candidates: tuple[Candidate, ...]
    selected: Candidate
    validation_blocks: tuple[Metrics, ...]
    test: Metrics
    test_blocks: tuple[Metrics, ...]


@dataclass(frozen=True)
class Floor:
    validation: Metrics
    validation_blocks: tuple[Metrics, ...]
    test: Metrics
    test_blocks: tuple[Metrics, ...]


@dataclass(frozen=True)
class Direction:
    against: str
    overall_mae: bool
    overall_rmse: bool
    mae_blocks: int
    rmse_blocks: int
    passed: bool


@dataclass(frozen=True)
class Gate:
    validation: tuple[Direction, ...]
    validation_passed: bool
    test: tuple[Direction, ...]
    test_passed: bool
    confirmed: bool


@dataclass(frozen=True)
class Observation:
    device: str
    nodes: int
    edges: int
    steps: int
    train_end: int
    validation_end: int
    lags: tuple[int, ...]
    sum_topology_int32: int
    floor: Floor
    structures: tuple[Structure, ...]
    gate: Gate


@dataclass(frozen=True, eq=False)
class _Selection:
    name: str
    covered_nodes: int
    sparse_calls: int
    aggregate: Tensor
    candidates: tuple[Candidate, ...]
    selected: Candidate
    validation_blocks: tuple[Metrics, ...]


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
    if not max(LAGS) < train_end < validation_end < len(signal):
        raise ValueError("expected ordered splits with enough training lag history")
    if min(validation_end - train_end, len(signal) - validation_end) < 3:
        raise ValueError("validation and test must each contain at least three rows")

    phase_mean, _ = _phase_mean(signal, train_end)
    baseline = Tensor.stack(
        *(phase_mean[(row + 1) % PERIOD] for row in range(len(signal)))
    ).realize()
    residual = (signal.y - baseline).realize()
    graph = signal.graph
    controls = (
        ("real", graph, residual),
        ("reverse", Graph(graph.nodes, graph.target, graph.source), residual),
        (
            "permuted",
            graph,
            residual[:, 1:].cat(residual[:, :1], dim=1).realize(),
        ),
    )

    floor_validation = _score(
        baseline[train_end:validation_end],
        signal.y[train_end:validation_end],
    )
    floor_validation_blocks = _blocks(
        baseline[train_end:validation_end],
        signal.y[train_end:validation_end],
    )
    selections = tuple(
        _select(
            name,
            structure,
            field,
            residual,
            baseline,
            signal.y,
            train_end=train_end,
            validation_end=validation_end,
        )
        for name, structure, field in controls
    )

    structures = tuple(
        _finish(selection, baseline, signal.y, validation_end, len(signal))
        for selection in selections
    )
    floor = Floor(
        floor_validation,
        floor_validation_blocks,
        _score(baseline[validation_end:], signal.y[validation_end:]),
        _blocks(baseline[validation_end:], signal.y[validation_end:]),
    )
    validation_gate = _directions(structures, floor, validation=True)
    test_gate = _directions(structures, floor, validation=False)
    validation_passed = all(direction.passed for direction in validation_gate)
    test_passed = all(direction.passed for direction in test_gate)
    return Observation(
        signal.x.device,
        graph.nodes,
        graph.edges,
        len(signal),
        train_end,
        validation_end,
        LAGS,
        2 * (graph.nodes + 1 + graph.edges),
        floor,
        structures,
        Gate(
            validation_gate,
            validation_passed,
            test_gate,
            test_passed,
            validation_passed and test_passed,
        ),
    )


def _select(
    name: str,
    graph: Graph,
    field: Tensor,
    residual: Tensor,
    baseline: Tensor,
    target: Tensor,
    *,
    train_end: int,
    validation_end: int,
) -> _Selection:
    aggregate = _incoming_mean(graph, field)
    sparse_calls = _sparse_calls(aggregate)
    aggregate.realize()
    covered_nodes = int((graph.in_degree(device=field.device) != 0).sum().item())
    candidates = []
    for lag in LAGS:
        alpha = _fit(aggregate, residual, lag, train_end)
        candidates.append(
            Candidate(
                lag,
                alpha,
                _score(
                    _predict(baseline, aggregate, lag, alpha, train_end, validation_end),
                    target[train_end:validation_end],
                ),
            )
        )
    candidates = tuple(candidates)
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.validation.rmse,
            candidate.validation.mae,
            candidate.lag,
        ),
    )
    prediction = _predict(
        baseline,
        aggregate,
        selected.lag,
        selected.alpha,
        train_end,
        validation_end,
    )
    return _Selection(
        name,
        covered_nodes,
        sparse_calls,
        aggregate,
        candidates,
        selected,
        _blocks(prediction, target[train_end:validation_end]),
    )


def _finish(
    selection: _Selection,
    baseline: Tensor,
    target: Tensor,
    start: int,
    end: int,
) -> Structure:
    prediction = _predict(
        baseline,
        selection.aggregate,
        selection.selected.lag,
        selection.selected.alpha,
        start,
        end,
    )
    return Structure(
        selection.name,
        selection.covered_nodes,
        selection.covered_nodes / target.shape[1],
        selection.sparse_calls,
        selection.candidates,
        selection.selected,
        selection.validation_blocks,
        _score(prediction, target[start:end]),
        _blocks(prediction, target[start:end]),
    )


def _incoming_mean(graph: Graph, field: Tensor) -> Tensor:
    return graph.mean(field)


def _fit(aggregate: Tensor, residual: Tensor, lag: int, train_end: int) -> float:
    signal = aggregate[:train_end - lag]
    target = residual[lag:train_end]
    numerator, denominator = (signal * target).sum(), signal.square().sum()
    Tensor.realize(numerator, denominator)
    scale = denominator.item()
    return 0.0 if scale == 0 else numerator.item() / scale


def _predict(
    baseline: Tensor,
    aggregate: Tensor,
    lag: int,
    alpha: float,
    start: int,
    end: int,
) -> Tensor:
    return baseline[start:end] + alpha * aggregate[start - lag:end - lag]


def _score(prediction: Tensor, target: Tensor) -> Metrics:
    error = prediction - target
    absolute, square = error.abs().sum(), error.square().sum()
    Tensor.realize(absolute, square)
    targets = target.numel()
    return Metrics(absolute.item() / targets, sqrt(square.item() / targets))


def _blocks(prediction: Tensor, target: Tensor) -> tuple[Metrics, ...]:
    size, extra = divmod(target.shape[0], 3)
    blocks = []
    start = 0
    for block in range(3):
        end = start + size + (1 if block < extra else 0)
        blocks.append(_score(prediction[start:end], target[start:end]))
        start = end
    return tuple(blocks)


def _directions(
    structures: tuple[Structure, ...],
    floor: Floor,
    *,
    validation: bool,
) -> tuple[Direction, ...]:
    real = structures[0]
    real_metrics = real.selected.validation if validation else real.test
    real_blocks = real.validation_blocks if validation else real.test_blocks
    comparisons = [
        (
            "floor",
            floor.validation if validation else floor.test,
            floor.validation_blocks if validation else floor.test_blocks,
        )
    ]
    comparisons.extend(
        (
            structure.name,
            structure.selected.validation if validation else structure.test,
            structure.validation_blocks if validation else structure.test_blocks,
        )
        for structure in structures[1:]
    )
    directions = []
    for name, metrics, blocks in comparisons:
        mae_blocks = sum(
            real_block.mae < other_block.mae
            for real_block, other_block in zip(real_blocks, blocks)
        )
        rmse_blocks = sum(
            real_block.rmse < other_block.rmse
            for real_block, other_block in zip(real_blocks, blocks)
        )
        mae, rmse = real_metrics.mae < metrics.mae, real_metrics.rmse < metrics.rmse
        directions.append(
            Direction(
                name,
                mae,
                rmse,
                mae_blocks,
                rmse_blocks,
                mae and rmse and mae_blocks >= 2 and rmse_blocks >= 2,
            )
        )
    return tuple(directions)


def _sparse_calls(tensor: Tensor) -> int:
    return sum(
        uop.src[0].arg.name == "csr_sum"
        for uop in tensor.uop.toposort()
        if uop.op is Ops.CALL
    )


def main() -> None:
    print(json.dumps(asdict(compare(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
