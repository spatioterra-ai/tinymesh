"""Test A3T-GCN against causal controls on METR-LA."""

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
from tinymesh.nn import A3TGCN


class Forecast:
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
            self.readout.weight = Tensor.zeros_like(self.readout.weight)
            assert self.readout.bias is not None
            self.readout.bias = Tensor.zeros_like(self.readout.bias)
        self.head = head

    def __call__(self, values: Tensor, graph: Graph, anchor: Tensor | None = None) -> Tensor:
        forecast = self.readout(self.encoder(values, graph).relu())
        if self.head == "direct":
            return forecast
        if anchor is None or anchor.shape != (*forecast.shape[:-1], 1):
            raise ValueError(f"residual anchor must have shape {(*forecast.shape[:-1], 1)}")
        return forecast + anchor


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


def observe(_device: str, *, history: int = 12, horizon: int = 12) -> ProtocolObservation:
    return observe_protocol(prepare(metr_la(device="CPU"), history=history, horizon=horizon))


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
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head, loss, evaluate_test)
    return train(
        prepare(metr_la(device="CPU"), history=history, horizon=horizon),
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
) -> SmokeObservation:
    _validate(("true",), (seed,), 1, batch_size, hidden_features, learning_rate, 1, head, loss)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    protocol = prepare(metr_la(device="CPU"), history=history, horizon=horizon)
    maximum = (protocol.train.windows + batch_size - 1) // batch_size
    if steps > maximum:
        raise ValueError(f"steps must not exceed the {maximum} training batches")

    tensors = _execution_tensors(protocol, device)
    Tensor.manual_seed(seed)
    model = Forecast(protocol.features.shape[2], hidden_features, history, horizon, head)
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
    train_step = _training_step(model, protocol.data.graph, optimizer, loss)
    losses = []
    start = perf_counter()
    for batch in islice(batches(protocol, protocol.train, batch_size, shuffle=seed, tensors=tensors), steps):
        batch = _execution_batch(batch, device, batch_size)
        losses.append(float(train_step(batch.values, batch.anchor, batch.target, batch.observed).item()))
    return SmokeObservation(
        device,
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
        _sparse_calls(model, protocol.data.graph, protocol, device),
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
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head, loss, evaluate_test)
    graphs, tensors, results = _graphs(protocol.data.graph), _execution_tensors(protocol, device), []
    for topology in topologies:
        for seed in seeds:
            Tensor.manual_seed(seed)
            model = Forecast(
                protocol.features.shape[2],
                hidden_features,
                protocol.train.history,
                protocol.train.horizon,
                head,
            )
            best_epoch, runtime, checkpoints, validation = _fit(
                model,
                graphs[topology],
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
                    _sparse_calls(model, graphs[topology], protocol, device),
                    runtime,
                    checkpoints,
                    validation,
                    (
                        _evaluate(
                            model,
                            graphs[topology],
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


def _fit(
    model: Forecast,
    graph: Graph,
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
    train_step = _training_step(model, graph, optimizer, loss)
    start = perf_counter()
    best_epoch = 0
    best = _evaluate(
        model,
        graph,
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
            graph,
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


def _training_step(model: Forecast, graph: Graph, optimizer, loss: str) -> TinyJit:
    @TinyJit
    @Context(TRAINING=1)
    def step(values: Tensor, anchor: Tensor, target: Tensor, observed: Tensor) -> Tensor:
        optimizer.zero_grad()
        objective = _objective(model(values, graph, anchor) - target, observed, loss).backward()
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
    model: Forecast,
    graph: Graph,
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
                predictors[execution.values.shape] = predict = _predictor(model, graph)
            prediction = predict(execution.values, execution.anchor)[:size].to(protocol.data.speed.device).realize()
            target = batch.target.to(protocol.data.speed.device).realize()
            yield protocol.standardizer.restore(prediction) - protocol.standardizer.restore(target), batch.observed.to(
                protocol.data.speed.device
            ).realize()

    return score(errors())


def _predictor(model: Forecast, graph: Graph):
    @TinyJit
    def predict(values: Tensor, anchor: Tensor) -> Tensor:
        return model(values, graph, anchor).realize()

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


def _snapshot(model: Forecast) -> dict[str, Tensor]:
    return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _parameter_count(model: Forecast) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _sparse_calls(model: Forecast, graph: Graph, protocol: Protocol, device: str) -> int:
    output = model(
        Tensor.zeros(1, protocol.train.history, graph.nodes, protocol.features.shape[2], device=device),
        graph,
        Tensor.zeros(1, graph.nodes, 1, device=device),
    )
    return sum(uop.src[0].arg.name == "csr_sum" for uop in output.uop.toposort() if uop.op is Ops.CALL)


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
    evaluate_test: bool = False,
) -> None:
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


def main() -> None:
    epochs, steps = getenv("EPOCHS", 0), getenv("STEPS", 0)
    if epochs and steps:
        raise SystemExit("EPOCHS and STEPS are mutually exclusive")
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
            **settings,
        )
    elif not epochs:
        observation = observe(Device.DEFAULT, **settings)
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
            evaluate_test=bool(getenv("TEST", 0)),
            **settings,
        )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
