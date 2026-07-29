"""Compare causal Montevideo forecasts in raw passenger-count units."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops

from experiments.directed_diffusion import DirectedDiffusion
from experiments.directed_gru import DiffusionForecast
from tinymesh import StaticGraphTemporalSignal
from tinymesh.datasets import MontevideoBus, montevideo_bus


class LocalForecast:
    def __init__(self, in_features: int, hidden_features: int) -> None:
        self.cell = nn.LSTMCell(in_features, hidden_features)
        self.readout = nn.Linear(hidden_features, 1)

    def __call__(self, values: Tensor, *, realize_steps: bool = False) -> Tensor:
        batch, _, nodes, features = values.shape
        state = None
        for step in range(values.shape[1]):
            state = self.cell(values[:, step].reshape(batch * nodes, features), state)
            if realize_steps:
                Tensor.realize(*state)
        return self.readout(state[0]).reshape(batch, nodes, 1)


Model = LocalForecast | DiffusionForecast


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


@dataclass(frozen=True)
class AffinityReport:
    name: str
    forward_changes_from_unit: int
    reverse_changes_from_unit: int


@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    metrics: Metrics


@dataclass(frozen=True)
class Result:
    model: str
    seed: int
    best_epoch: int
    parameters: int
    sparse_calls: int
    runtime_seconds: float
    checkpoints: tuple[Checkpoint, ...]
    validation: Metrics
    test: Metrics


@dataclass(frozen=True)
class ForecastObservation:
    device: str
    epochs: int
    history: int
    batch_size: int
    hidden_features: int
    learning_rate: float
    checkpoint_every: int
    isolated_out: int
    isolated_in: int
    protocol: Observation
    affinity: tuple[AffinityReport, ...]
    results: tuple[Result, ...]


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
    return _protocol(
        device,
        signal,
        splits,
        input_standardizer,
        target_standardizer,
        history=history,
        batch_size=batch_size,
    )


def forecast(
    device: str,
    *,
    models: tuple[str, ...] = ("lstm", "unit", "coordinate", "road"),
    seeds: tuple[int, ...] = (0, 1, 2),
    epochs: int = 10,
    history: int = 24,
    batch_size: int = 32,
    hidden_features: int = 4,
    learning_rate: float = 0.01,
    checkpoint_every: int = 5,
) -> ForecastObservation:
    allowed = {"lstm", "unit", "coordinate", "road"}
    if not models or any(name not in allowed for name in models):
        raise ValueError(f"models must be drawn from {sorted(allowed)}")
    if not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    for name, value in (
        ("epochs", epochs),
        ("history", history),
        ("batch_size", batch_size),
        ("hidden_features", hidden_features),
        ("checkpoint_every", checkpoint_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    data = montevideo_bus(lags=1, device=device)
    raw = split_signal(data.signal, history=history)
    normalized, input_standardizer, target_standardizer = normalize_splits(raw)
    protocol = _protocol(
        device,
        data.signal,
        raw,
        input_standardizer,
        target_standardizer,
        history=history,
        batch_size=batch_size,
    )
    diffusion, affinity = _diffusions(data)
    results = []
    for name in models:
        for seed in seeds:
            Tensor.manual_seed(seed)
            model = _model(name, hidden_features)
            operator = None if name == "lstm" else diffusion[name]
            best_epoch, runtime, checkpoints, validation = _fit(
                model,
                operator,
                normalized.train,
                normalized.validation,
                target_standardizer,
                epochs=epochs,
                batch_size=batch_size,
                history=history,
                learning_rate=learning_rate,
                checkpoint_every=checkpoint_every,
            )
            results.append(
                Result(
                    name,
                    seed,
                    best_epoch,
                    _parameter_count(model),
                    _sparse_calls(model, operator, data.signal.graph.nodes, device),
                    runtime,
                    checkpoints,
                    validation,
                    _evaluate(
                        model,
                        operator,
                        normalized.test,
                        target_standardizer,
                        history=history,
                    ),
                )
            )
    return ForecastObservation(
        device,
        epochs,
        history,
        batch_size,
        hidden_features,
        learning_rate,
        checkpoint_every,
        data.signal.graph.nodes - len(set(data.signal.graph.source)),
        data.signal.graph.nodes - len(set(data.signal.graph.target)),
        protocol,
        affinity,
        tuple(results),
    )


def _protocol(
    device: str,
    signal: StaticGraphTemporalSignal,
    splits: SignalSplits,
    input_standardizer: Standardizer,
    target_standardizer: Standardizer,
    *,
    history: int,
    batch_size: int,
) -> Observation:
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


def _diffusions(
    data: MontevideoBus,
) -> tuple[dict[str, DirectedDiffusion], tuple[AffinityReport, ...]]:
    graph = data.signal.graph
    source = graph.edge_values(data.position, endpoint="source")
    target = graph.edge_values(data.position, endpoint="target")
    coordinate_distance = ((target - source) ** 2).sum(axis=-1).sqrt().realize()
    affinities = {
        "unit": Tensor.ones(
            graph.edges,
            dtype=data.signal.x.dtype,
            device=data.signal.x.device,
        ).realize(),
        "coordinate": (1 / coordinate_distance).realize(),
        "road": (1 / data.road_distance).realize(),
    }
    diffusion = {
        name: DirectedDiffusion(graph, affinity)
        for name, affinity in affinities.items()
    }
    weights = [
        weight
        for operator in diffusion.values()
        for weight in (operator.forward_weight, operator.reverse_weight)
    ]
    Tensor.realize(*weights)
    unit = diffusion["unit"]
    report = tuple(
        AffinityReport(
            name,
            _changes(operator.forward_weight, unit.forward_weight),
            _changes(operator.reverse_weight, unit.reverse_weight),
        )
        for name, operator in diffusion.items()
    )
    return diffusion, report


def _changes(values: Tensor, unit: Tensor) -> int:
    return int(((values - unit).abs() > 1e-6).sum().item())


def _model(name: str, hidden_features: int) -> Model:
    if name == "lstm":
        return LocalForecast(1, hidden_features)
    return DiffusionForecast(1, hidden_features)


def _fit(
    model: Model,
    diffusion: DirectedDiffusion | None,
    train: StaticGraphTemporalSignal,
    validation: StaticGraphTemporalSignal,
    target_standardizer: Standardizer,
    *,
    epochs: int,
    batch_size: int,
    history: int,
    learning_rate: float,
    checkpoint_every: int,
) -> tuple[int, float, tuple[Checkpoint, ...], Metrics]:
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
            loss = (_predict(model, values, diffusion) - target).square().mean().backward()
            return loss.realize(*optimizer.schedule_step())

        return step

    start = perf_counter()
    best_epoch = 0
    best_metrics = _evaluate(
        model,
        diffusion,
        validation,
        target_standardizer,
        history=history,
    )
    best_error = best_metrics.rmse
    best_state = _snapshot(model)
    checkpoints = [Checkpoint(0, best_metrics)]
    steps: dict[tuple[int, ...], TinyJit] = {}
    for epoch in range(1, epochs + 1):
        for values, target in train.batches(
            batch_size=batch_size,
            history=history,
        ):
            step = steps.get(values.shape)
            if step is None:
                steps[values.shape] = step = make_step()
            step(values.contiguous(), target.contiguous())
        if epoch % checkpoint_every != 0 and epoch != epochs:
            continue
        validation_metrics = _evaluate(
            model,
            diffusion,
            validation,
            target_standardizer,
            history=history,
        )
        checkpoints.append(Checkpoint(epoch, validation_metrics))
        if validation_metrics.rmse < best_error:
            best_epoch = epoch
            best_error = validation_metrics.rmse
            best_metrics = validation_metrics
            best_state = _snapshot(model)
    runtime = perf_counter() - start
    nn.state.load_state_dict(model, best_state, verbose=False)
    return best_epoch, runtime, tuple(checkpoints), best_metrics


def _evaluate(
    model: Model,
    diffusion: DirectedDiffusion | None,
    signal: StaticGraphTemporalSignal,
    target_standardizer: Standardizer,
    *,
    history: int,
) -> Metrics:
    absolute = square = 0.0
    targets = 0
    for values, target in signal.batches(
        batch_size=_windows(signal, history),
        history=history,
    ):
        prediction = target_standardizer.restore(
            _predict(model, values, diffusion, realize_steps=True)
        )
        raw_target = target_standardizer.restore(target)
        error = prediction - raw_target
        absolute_error, squared_error = error.abs().sum(), error.square().sum()
        Tensor.realize(absolute_error, squared_error)
        absolute += absolute_error.item()
        square += squared_error.item()
        targets += target.numel()
    return Metrics(absolute / targets, sqrt(square / targets))


def _predict(
    model: Model,
    values: Tensor,
    diffusion: DirectedDiffusion | None,
    *,
    realize_steps: bool = False,
) -> Tensor:
    if isinstance(model, LocalForecast):
        return model(values, realize_steps=realize_steps)
    if diffusion is None:
        raise ValueError("diffusion model requires one operator")
    return model(values, diffusion, realize_steps=realize_steps)


def _snapshot(model: Model) -> dict[str, Tensor]:
    return {
        name: value.detach().clone().realize()
        for name, value in nn.state.get_state_dict(model).items()
    }


def _parameter_count(model: Model) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _sparse_calls(
    model: Model,
    diffusion: DirectedDiffusion | None,
    nodes: int,
    device: str,
) -> int:
    output = _predict(
        model,
        Tensor.zeros(1, 1, nodes, 1, device=device),
        diffusion,
    )
    return sum(
        uop.src[0].arg.name == "csr_sum"
        for uop in output.uop.toposort()
        if uop.op is Ops.CALL
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
    epochs = getenv("EPOCHS", 0)
    model = getenv("MODEL", "all")
    seed = getenv("SEED", -1)
    settings = {
        "history": getenv("HISTORY", 24),
        "batch_size": getenv("BS", 32),
    }
    observation = (
        compare(Device.DEFAULT, **settings)
        if epochs == 0
        else forecast(
            Device.DEFAULT,
            models=("lstm", "unit", "coordinate", "road")
            if model == "all" else (model,),
            seeds=(0, 1, 2) if seed < 0 else (seed,),
            epochs=epochs,
            hidden_features=getenv("HIDDEN", 4),
            learning_rate=getenv("LR", 0.01),
            checkpoint_every=getenv("CHECKPOINT_EVERY", 5),
            **settings,
        )
    )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
