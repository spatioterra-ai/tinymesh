"""Test temporal graph models against causal controls on METR-LA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import islice
from random import Random
from time import perf_counter

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops

from experiments.metr_la_protocol import (
    Protocol,
    ProtocolObservation,
    Scores,
    WindowBatch,
    batches,
    observe as observe_protocol,
    prepare,
    score,
)
from tinymesh import Graph
from tinymesh.datasets import metr_la
from tinymesh.nn import A3TGCN, DiffusionGRU, DirectedDiffusion


class A3Forecast:
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        periods: int,
        horizon: int,
        head: str = "direct",
    ) -> None:
        if head not in ("direct", "residual"):
            raise ValueError("head must be 'direct' or 'residual'")
        self.encoder = A3TGCN(in_features, hidden_features, periods)
        self.readout = nn.Linear(hidden_features, horizon)
        if head == "residual":
            _zero(self.readout)
        self.head = head

    def __call__(self, values: Tensor, graph: Graph, anchor: Tensor | None = None) -> Tensor:
        return _output(self.readout(self.encoder(values, graph).relu()), anchor, self.head)


class DiffusionForecast:
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        periods: int,
        horizon: int,
        head: str = "direct",
    ) -> None:
        if head not in ("direct", "residual"):
            raise ValueError("head must be 'direct' or 'residual'")
        self.cell = DiffusionGRU(in_features, hidden_features)
        self.readout = nn.Linear(hidden_features, horizon)
        if head == "residual":
            _zero(self.readout)
        self.in_features, self.periods, self.head = in_features, periods, head

    def __call__(
        self,
        values: Tensor,
        diffusion: DirectedDiffusion,
        anchor: Tensor | None = None,
    ) -> Tensor:
        expected = (self.periods, diffusion.graph.nodes, self.in_features)
        if values.ndim < 3 or values.shape[-3:] != expected:
            raise ValueError(
                f"values must have shape [..., {self.periods}, "
                f"{diffusion.graph.nodes}, {self.in_features}], got {values.shape}"
            )
        hidden = None
        for period in range(self.periods):
            hidden = self.cell(values[..., period, :, :], diffusion, hidden)
        return _output(self.readout(hidden.relu()), anchor, self.head)


@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    validation: Scores


@dataclass(frozen=True)
class Result:
    topology: str
    seed: int
    best_epoch: int
    parameters: int
    sparse_calls: int
    runtime_seconds: float
    checkpoints: tuple[Checkpoint, ...]
    validation: Scores
    test: Scores | None


@dataclass(frozen=True)
class ForecastObservation:
    device: str
    architecture: str
    head: str
    loss: str
    evaluate_test: bool
    epochs: int
    batch_size: int
    hidden_features: int
    learning_rate: float
    checkpoint_every: int
    protocol: ProtocolObservation
    results: tuple[Result, ...]


@dataclass(frozen=True)
class SmokeObservation:
    device: str
    architecture: str
    head: str
    loss: str
    seed: int
    steps: int
    trained_windows: int
    batch_size: int
    hidden_features: int
    history: int
    horizon: int
    nodes: int
    edges: int
    parameters: int
    sparse_calls: int
    first_loss: float
    last_loss: float
    runtime_seconds: float


def observe(
    _device: str,
    *,
    history: int = 12,
    horizon: int = 12,
    architecture: str = "a3tgcn",
) -> ProtocolObservation:
    _validate_architecture(architecture)
    return observe_protocol(prepare(
        metr_la(device="CPU"),
        history=history,
        horizon=horizon,
        feature_set=_feature_set(architecture),
    ))


def forecast(
    device: str,
    *,
    topologies: tuple[str, ...] = ("true", "permuted", "self"),
    seeds: tuple[int, ...] = (0, 1, 2),
    epochs: int = 30,
    history: int = 12,
    horizon: int = 12,
    batch_size: int = 32,
    hidden_features: int = 32,
    learning_rate: float = 0.001,
    checkpoint_every: int = 5,
    head: str = "direct",
    loss: str = "mse",
    evaluate_test: bool = False,
    architecture: str = "a3tgcn",
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head, loss, architecture, evaluate_test)
    return train(
        prepare(
            metr_la(device="CPU"),
            history=history,
            horizon=horizon,
            feature_set=_feature_set(architecture),
        ),
        device=device,
        topologies=topologies,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        hidden_features=hidden_features,
        learning_rate=learning_rate,
        checkpoint_every=checkpoint_every,
        head=head,
        loss=loss,
        evaluate_test=evaluate_test,
        architecture=architecture,
    )


def smoke(
    device: str,
    *,
    steps: int,
    seed: int = 0,
    history: int = 12,
    horizon: int = 12,
    batch_size: int = 32,
    hidden_features: int = 32,
    learning_rate: float = 0.001,
    head: str = "direct",
    loss: str = "mse",
    architecture: str = "a3tgcn",
) -> SmokeObservation:
    _validate(("true",), (seed,), 1, batch_size, hidden_features, learning_rate, 1, head, loss, architecture)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    protocol = prepare(
        metr_la(device="CPU"),
        history=history,
        horizon=horizon,
        feature_set=_feature_set(architecture),
    )
    maximum = (protocol.train.windows + batch_size - 1) // batch_size
    if steps > maximum:
        raise ValueError(f"steps must not exceed the {maximum} training batches")

    tensors = _execution_tensors(protocol, device)
    Tensor.manual_seed(seed)
    model = _model(architecture, protocol, hidden_features, head)
    operator = _operators(protocol, architecture, device)["true"]
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
    train_step = _training_step(model, operator, optimizer, loss)
    losses = []
    start = perf_counter()
    for batch in islice(batches(protocol, protocol.train, batch_size, shuffle=seed, tensors=tensors), steps):
        batch = _execution_batch(batch, device, batch_size)
        losses.append(float(train_step(batch.values, batch.anchor, batch.target, batch.observed).item()))
    return SmokeObservation(
        device,
        architecture,
        head,
        loss,
        seed,
        steps,
        min(steps * batch_size, protocol.train.windows),
        batch_size,
        hidden_features,
        history,
        horizon,
        protocol.data.graph.nodes,
        protocol.data.graph.edges,
        _parameter_count(model),
        _sparse_calls(model, operator, protocol, device),
        losses[0],
        losses[-1],
        perf_counter() - start,
    )


def train(
    protocol: Protocol,
    *,
    device: str = Device.DEFAULT,
    topologies: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    hidden_features: int,
    learning_rate: float,
    checkpoint_every: int,
    head: str = "direct",
    loss: str = "mse",
    evaluate_test: bool = False,
    architecture: str = "a3tgcn",
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head, loss, architecture, evaluate_test)
    if protocol.feature_set != _feature_set(architecture):
        raise ValueError(f"{architecture} requires the {_feature_set(architecture)!r} feature set")
    operators, tensors, results = _operators(protocol, architecture, device), _execution_tensors(protocol, device), []
    for topology in topologies:
        for seed in seeds:
            Tensor.manual_seed(seed)
            model = _model(architecture, protocol, hidden_features, head)
            best_epoch, runtime, checkpoints, validation = _fit(
                model,
                operators[topology],
                protocol,
                tensors,
                device=device,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                checkpoint_every=checkpoint_every,
                loss=loss,
            )
            results.append(
                Result(
                    topology,
                    seed,
                    best_epoch,
                    _parameter_count(model),
                    _sparse_calls(model, operators[topology], protocol, device),
                    runtime,
                    checkpoints,
                    validation,
                    (
                        _evaluate(
                            model,
                            operators[topology],
                            protocol,
                            protocol.test,
                            tensors,
                            device=device,
                            batch_size=batch_size,
                        )
                        if evaluate_test
                        else None
                    ),
                )
            )
    return ForecastObservation(
        device,
        architecture,
        head,
        loss,
        evaluate_test,
        epochs,
        batch_size,
        hidden_features,
        learning_rate,
        checkpoint_every,
        observe_protocol(protocol),
        tuple(results),
    )


def _graphs(graph: Graph) -> dict[str, Graph]:
    permutation = list(range(graph.nodes))
    Random(0).shuffle(permutation)
    return {
        "true": graph,
        "permuted": Graph(
            graph.nodes,
            [permutation[source] for source in graph.source],
            [permutation[target] for target in graph.target],
        ),
        "self": Graph(graph.nodes, list(range(graph.nodes)), list(range(graph.nodes))),
    }


def _operators(
    protocol: Protocol,
    architecture: str,
    device: str,
) -> dict[str, Graph | DirectedDiffusion]:
    graphs = _graphs(protocol.data.graph)
    if architecture == "a3tgcn":
        return graphs
    affinity = protocol.data.affinity.to(device).realize()
    operators = {
        "true": DirectedDiffusion(graphs["true"], affinity),
        "permuted": DirectedDiffusion(graphs["permuted"], affinity),
        "self": DirectedDiffusion(
            graphs["self"],
            Tensor.ones(graphs["self"].edges, dtype=affinity.dtype, device=device),
        ),
    }
    Tensor.realize(*(
        weight
        for operator in operators.values()
        for weight in (operator.forward_weight, operator.reverse_weight)
    ))
    return operators


def _model(
    architecture: str,
    protocol: Protocol,
    hidden_features: int,
    head: str,
) -> A3Forecast | DiffusionForecast:
    model = A3Forecast if architecture == "a3tgcn" else DiffusionForecast
    return model(
        protocol.features.shape[2],
        hidden_features,
        protocol.train.history,
        protocol.train.horizon,
        head,
    )


def _fit(
    model: A3Forecast | DiffusionForecast,
    operator: Graph | DirectedDiffusion,
    protocol: Protocol,
    tensors: tuple[Tensor, Tensor, Tensor],
    *,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_every: int,
    loss: str,
) -> tuple[int, float, tuple[Checkpoint, ...], Scores]:
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
    train_step = _training_step(model, operator, optimizer, loss)
    start = perf_counter()
    best_epoch = 0
    best = _evaluate(
        model,
        operator,
        protocol,
        protocol.validation,
        tensors,
        device=device,
        batch_size=batch_size,
    )
    best_state, checkpoints = _snapshot(model), [Checkpoint(0, best)]
    for epoch in range(1, epochs + 1):
        for batch in batches(
            protocol,
            protocol.train,
            batch_size,
            shuffle=seed * 1_000_003 + epoch,
            tensors=tensors,
        ):
            batch = _execution_batch(batch, device, batch_size)
            train_step(batch.values, batch.anchor, batch.target, batch.observed)
        if epoch % checkpoint_every != 0 and epoch != epochs:
            continue
        validation = _evaluate(
            model,
            operator,
            protocol,
            protocol.validation,
            tensors,
            device=device,
            batch_size=batch_size,
        )
        checkpoints.append(Checkpoint(epoch, validation))
        if validation.overall.rmse < best.overall.rmse:
            best_epoch, best, best_state = epoch, validation, _snapshot(model)
    runtime = perf_counter() - start
    nn.state.load_state_dict(model, best_state, verbose=False)
    return best_epoch, runtime, tuple(checkpoints), best


def _training_step(
    model: A3Forecast | DiffusionForecast,
    operator: Graph | DirectedDiffusion,
    optimizer,
    loss: str,
) -> TinyJit:
    @TinyJit
    @Context(TRAINING=1)
    def step(values: Tensor, anchor: Tensor, target: Tensor, observed: Tensor) -> Tensor:
        optimizer.zero_grad()
        objective = _objective(model(values, operator, anchor) - target, observed, loss).backward()
        return objective.realize(*optimizer.schedule_step())

    return step


def _objective(error: Tensor, observed: Tensor, loss: str) -> Tensor:
    absolute = error.abs()
    if loss == "mse":
        element = error.square()
    elif loss == "mae":
        element = absolute
    else:
        element = (absolute < 1).where(error.square() / 2, absolute - 0.5)
    weight = observed.cast(error.dtype)
    return (element * weight).sum() / weight.sum()


def _evaluate(
    model: A3Forecast | DiffusionForecast,
    operator: Graph | DirectedDiffusion,
    protocol: Protocol,
    span,
    tensors: tuple[Tensor, Tensor, Tensor],
    *,
    device: str,
    batch_size: int,
) -> Scores:
    predictors = {}  # TinyJit captures parameter buffers, so one evaluator owns its lifetime.

    def errors():
        for batch in batches(protocol, span, batch_size, tensors=tensors):
            size = len(batch.starts)
            execution = _execution_batch(batch, device, batch_size)
            predict = predictors.get(execution.values.shape)
            if predict is None:
                predictors[execution.values.shape] = predict = _predictor(model, operator)
            prediction = predict(execution.values, execution.anchor)[:size].to(protocol.data.speed.device).realize()
            target = batch.target.to(protocol.data.speed.device).realize()
            yield protocol.standardizer.restore(prediction) - protocol.standardizer.restore(target), batch.observed.to(
                protocol.data.speed.device
            ).realize()

    return score(errors())


def _predictor(
    model: A3Forecast | DiffusionForecast,
    operator: Graph | DirectedDiffusion,
):
    @TinyJit
    def predict(values: Tensor, anchor: Tensor) -> Tensor:
        return model(values, operator, anchor).realize()

    return predict


def _execution_tensors(protocol: Protocol, device: str) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(value.to(device).realize() for value in (protocol.features, protocol.target, protocol.observed))


def _execution_batch(batch: WindowBatch, device: str, batch_size: int) -> WindowBatch:
    if len(batch.starts) < batch_size:
        padding = batch_size - len(batch.starts)
        batch = WindowBatch(
            batch.values.cat(batch.values[-1:].expand(padding, *batch.values.shape[1:]), dim=0),
            batch.anchor.cat(batch.anchor[-1:].expand(padding, *batch.anchor.shape[1:]), dim=0),
            batch.target.cat(batch.target[-1:].expand(padding, *batch.target.shape[1:]), dim=0),
            batch.observed.cat(
                Tensor.zeros(padding, *batch.observed.shape[1:], dtype=batch.observed.dtype, device=batch.observed.device),
                dim=0,
            ),
            batch.starts + (batch.starts[-1],) * padding,
        )
    values = tuple(
        value.contiguous().to(device).realize()
        for value in (batch.values, batch.anchor, batch.target, batch.observed)
    )
    return WindowBatch(*values, batch.starts)


def _snapshot(model: A3Forecast | DiffusionForecast) -> dict[str, Tensor]:
    return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _parameter_count(model: A3Forecast | DiffusionForecast) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _sparse_calls(
    model: A3Forecast | DiffusionForecast,
    operator: Graph | DirectedDiffusion,
    protocol: Protocol,
    device: str,
) -> int:
    output = model(
        Tensor.zeros(1, protocol.train.history, protocol.data.graph.nodes, protocol.features.shape[2], device=device),
        operator,
        Tensor.zeros(1, protocol.data.graph.nodes, 1, device=device),
    )
    return sum(uop.src[0].arg.name == "csr_sum" for uop in output.uop.toposort() if uop.op is Ops.CALL)


def _zero(readout: nn.Linear) -> None:
    readout.weight = Tensor.zeros_like(readout.weight)
    assert readout.bias is not None
    readout.bias = Tensor.zeros_like(readout.bias)


def _output(forecast: Tensor, anchor: Tensor | None, head: str) -> Tensor:
    if head == "direct":
        return forecast
    if anchor is None or anchor.shape != (*forecast.shape[:-1], 1):
        raise ValueError(f"residual anchor must have shape {(*forecast.shape[:-1], 1)}")
    return forecast + anchor


def _feature_set(architecture: str) -> str:
    return "linear_time" if architecture == "a3tgcn" else "calendar"


def _validate_architecture(architecture: str) -> None:
    if architecture not in ("a3tgcn", "diffusion_gru"):
        raise ValueError("architecture must be 'a3tgcn' or 'diffusion_gru'")


def _validate(
    topologies: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    hidden_features: int,
    learning_rate: float,
    checkpoint_every: int,
    head: str,
    loss: str,
    architecture: str,
    evaluate_test: bool = False,
) -> None:
    _validate_architecture(architecture)
    allowed = {"true", "permuted", "self"}
    if not topologies or any(topology not in allowed for topology in topologies):
        raise ValueError(f"topologies must be drawn from {sorted(allowed)}")
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("hidden_features", hidden_features),
        ("checkpoint_every", checkpoint_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if head not in ("direct", "residual"):
        raise ValueError("head must be 'direct' or 'residual'")
    if loss not in ("mse", "mae", "huber"):
        raise ValueError("loss must be 'mse', 'mae', or 'huber'")
    if not isinstance(evaluate_test, bool):
        raise ValueError("evaluate_test must be boolean")


def run(architecture: str = "a3tgcn") -> ProtocolObservation | ForecastObservation | SmokeObservation:
    _validate_architecture(architecture)
    epochs, steps, evaluate_test = getenv("EPOCHS", 0), getenv("STEPS", 0), getenv("TEST", 0)
    if epochs and steps:
        raise SystemExit("EPOCHS and STEPS are mutually exclusive")
    if evaluate_test not in (0, 1):
        raise SystemExit("TEST must be 0 or 1")
    settings = {"history": getenv("HISTORY", 12), "horizon": getenv("HORIZON", 12)}
    if steps:
        observation = smoke(
            Device.DEFAULT,
            steps=steps,
            seed=max(0, getenv("SEED", -1)),
            batch_size=getenv("BS", 32),
            hidden_features=getenv("HIDDEN", 32),
            learning_rate=getenv("LR", 0.001),
            head=getenv("HEAD", "direct"),
            loss=getenv("LOSS", "mse"),
            architecture=architecture,
            **settings,
        )
    elif not epochs:
        observation = observe(Device.DEFAULT, architecture=architecture, **settings)
    else:
        topology, seed = getenv("MODEL", "all"), getenv("SEED", -1)
        observation = forecast(
            Device.DEFAULT,
            topologies=("true", "permuted", "self") if topology == "all" else (topology,),
            seeds=(0, 1, 2) if seed < 0 else (seed,),
            epochs=epochs,
            batch_size=getenv("BS", 32),
            hidden_features=getenv("HIDDEN", 32),
            learning_rate=getenv("LR", 0.001),
            checkpoint_every=getenv("CHECKPOINT_EVERY", 5),
            head=getenv("HEAD", "direct"),
            loss=getenv("LOSS", "mse"),
            evaluate_test=bool(evaluate_test),
            architecture=architecture,
            **settings,
        )
    return observation


def main() -> None:
    print(json.dumps(asdict(run()), indent=2))


if __name__ == "__main__":
    main()
