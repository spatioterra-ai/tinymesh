"""Train fixed-graph recurrent models on the chickenpox signal."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from tinymesh import Graph, StaticGraphTemporalSignal
from tinymesh.datasets import chickenpox
from tinymesh.nn import GConvGRU, TGCN


class Forecast:
    def __init__(self, cell: TGCN | GConvGRU, hidden_features: int) -> None:
        self.cell = cell
        self.readout = nn.Linear(hidden_features, 1)

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        hidden = None
        for step in range(values.shape[1]):
            hidden = self.cell(values[:, step], graph, hidden)
        return self.readout(hidden)


class LSTMForecast:
    def __init__(self, hidden_features: int) -> None:
        self.cell = nn.LSTMCell(1, hidden_features)
        self.readout = nn.Linear(hidden_features, 1)

    def __call__(self, values: Tensor, _graph: Graph) -> Tensor:
        batch, _, nodes, features = values.shape
        state = None
        for step in range(values.shape[1]):
            state = self.cell(values[:, step].reshape(batch * nodes, features), state)
        return self.readout(state[0]).reshape(batch, nodes, 1)


Model = Forecast | LSTMForecast


@dataclass(frozen=True)
class Metrics:
    mse: float
    mae: float


@dataclass(frozen=True)
class Result:
    model: str
    parameters: int
    train: Metrics
    validation: Metrics
    test: Metrics


@dataclass(frozen=True)
class Observation:
    device: str
    seed: int
    epochs: int
    history: int
    batch_size: int
    hidden_features: int
    learning_rate: float
    source_edges: int
    chebyshev_edges: int
    train_windows: int
    validation_windows: int
    test_windows: int
    climatology: Metrics
    persistence: Metrics
    models: tuple[Result, ...]


def compare(
    device: str,
    *,
    seed: int,
    epochs: int,
    history: int,
    batch_size: int,
    hidden_features: int,
    learning_rate: float,
) -> Observation:
    signal = chickenpox(lags=1, device=device)
    development, test = signal.split(0.8)
    train, validation = development.split(0.875)
    chebyshev_graph = _without_self_loops(signal.graph)
    settings = {"batch_size": batch_size, "history": history}

    results = []
    for name in ("lstm", "tgcn", "gconv_gru"):
        Tensor.manual_seed(seed)
        model = _model(name, hidden_features)
        graph = signal.graph if name == "tgcn" else chebyshev_graph
        _fit(model, graph, train, epochs=epochs, learning_rate=learning_rate, **settings)
        results.append(
            Result(
                model=name,
                parameters=_parameter_count(model),
                train=_evaluate(model, graph, train, **settings),
                validation=_evaluate(model, graph, validation, **settings),
                test=_evaluate(model, graph, test, **settings),
            )
        )

    return Observation(
        device=device,
        seed=seed,
        epochs=epochs,
        history=history,
        batch_size=batch_size,
        hidden_features=hidden_features,
        learning_rate=learning_rate,
        source_edges=signal.graph.edges,
        chebyshev_edges=chebyshev_graph.edges,
        train_windows=_windows(train, history),
        validation_windows=_windows(validation, history),
        test_windows=_windows(test, history),
        climatology=_baseline(test, persistence=False, **settings),
        persistence=_baseline(test, persistence=True, **settings),
        models=tuple(results),
    )


def _fit(
    model: Model,
    graph: Graph,
    signal: StaticGraphTemporalSignal,
    *,
    epochs: int,
    batch_size: int,
    history: int,
    learning_rate: float,
) -> None:
    optimizer = nn.optim.Adam(
        nn.state.get_parameters(model),
        lr=learning_rate,
        fused=False,
    )

    def make_step() -> TinyJit:
        @TinyJit
        @Context(TRAINING=1)
        def step(values: Tensor, target: Tensor) -> Tensor:
            optimizer.zero_grad()
            loss = (model(values, graph) - target).square().mean().backward()
            return loss.realize(*optimizer.schedule_step())

        return step

    steps: dict[tuple[int, ...], TinyJit] = {}
    for _ in range(epochs):
        for values, target in signal.batches(batch_size=batch_size, history=history):
            step = steps.get(values.shape)
            if step is None:
                steps[values.shape] = step = make_step()
            step(values.contiguous(), target.contiguous())


def _evaluate(
    model: Model,
    graph: Graph,
    signal: StaticGraphTemporalSignal,
    *,
    batch_size: int,
    history: int,
) -> Metrics:
    return _metrics(
        (model(values, graph), target)
        for values, target in signal.batches(batch_size=batch_size, history=history)
    )


def _baseline(
    signal: StaticGraphTemporalSignal,
    *,
    batch_size: int,
    history: int,
    persistence: bool,
) -> Metrics:
    return _metrics(
        (values[:, -1] if persistence else target * 0, target)
        for values, target in signal.batches(batch_size=batch_size, history=history)
    )


def _metrics(predictions: Iterable[tuple[Tensor, Tensor]]) -> Metrics:
    square, absolute, count = 0.0, 0.0, 0
    for prediction, target in predictions:
        error = prediction - target
        squared_error, absolute_error = error.square().sum(), error.abs().sum()
        squared_error.realize(absolute_error)
        square += squared_error.item()
        absolute += absolute_error.item()
        count += target.numel()
    return Metrics(mse=square / count, mae=absolute / count)


def _without_self_loops(graph: Graph) -> Graph:
    edges = [
        (source, target)
        for source, target in zip(graph.source, graph.target)
        if source != target
    ]
    return Graph(
        graph.nodes,
        [source for source, _ in edges],
        [target for _, target in edges],
    )


def _windows(signal: StaticGraphTemporalSignal, history: int) -> int:
    return len(signal) - history + 1


def _model(name: str, hidden_features: int) -> Model:
    if name == "lstm":
        return LSTMForecast(hidden_features)
    if name == "tgcn":
        return Forecast(TGCN(1, hidden_features), hidden_features)
    if name == "gconv_gru":
        return Forecast(GConvGRU(1, hidden_features, 2), hidden_features)
    raise ValueError(name)


def _parameter_count(model: Model) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def main() -> None:
    observation = compare(
        Device.DEFAULT,
        seed=getenv("SEED", 0),
        epochs=getenv("EPOCHS", 50),
        history=getenv("HISTORY", 8),
        batch_size=getenv("BS", 357),
        hidden_features=getenv("HIDDEN", 4),
        learning_rate=getenv("LR", 0.01),
    )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
