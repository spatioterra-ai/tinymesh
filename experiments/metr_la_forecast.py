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

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        forecast = self.readout(self.encoder(values, graph).relu())
        return forecast if self.head == "direct" else forecast + values[..., -1, :, :1]


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
    test: Scores


@dataclass(frozen=True)
class ForecastObservation:
    device: str
    head: str
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
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head)
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
) -> SmokeObservation:
    _validate(("true",), (seed,), 1, batch_size, hidden_features, learning_rate, 1, head)
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
    train_step = _training_step(model, protocol.data.graph, optimizer)
    losses = []
    start = perf_counter()
    for batch in islice(batches(protocol, protocol.train, batch_size, shuffle=seed, tensors=tensors), steps):
        batch = _execution_batch(batch, device, batch_size)
        losses.append(float(train_step(batch.values, batch.target, batch.observed).item()))
    return SmokeObservation(
        device,
        head,
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
) -> ForecastObservation:
    _validate(topologies, seeds, epochs, batch_size, hidden_features, learning_rate, checkpoint_every, head)
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
                    _evaluate(
                        model,
                        graphs[topology],
                        protocol,
                        protocol.test,
                        tensors,
                        device=device,
                        batch_size=batch_size,
                    ),
                )
            )
    return ForecastObservation(
        device,
        head,
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
) -> tuple[int, float, tuple[Checkpoint, ...], Scores]:
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
    train_step = _training_step(model, graph, optimizer)
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
            train_step(batch.values, batch.target, batch.observed)
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


def _training_step(model: Forecast, graph: Graph, optimizer) -> TinyJit:
    @TinyJit
    @Context(TRAINING=1)
    def step(values: Tensor, target: Tensor, observed: Tensor) -> Tensor:
        optimizer.zero_grad()
        weight = observed.cast(target.dtype)
        error = (model(values, graph) - target) * weight
        loss = (error.square().sum() / weight.sum()).backward()
        return loss.realize(*optimizer.schedule_step())

    return step


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
            prediction = predict(execution.values)[:size].to(protocol.data.speed.device).realize()
            target = batch.target.to(protocol.data.speed.device).realize()
            yield protocol.standardizer.restore(prediction) - protocol.standardizer.restore(target), batch.observed.to(
                protocol.data.speed.device
            ).realize()

    return score(errors())


def _predictor(model: Forecast, graph: Graph):
    @TinyJit
    def predict(values: Tensor) -> Tensor:
        return model(values, graph).realize()

    return predict


def _execution_tensors(protocol: Protocol, device: str) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(value.to(device).realize() for value in (protocol.features, protocol.target, protocol.observed))


def _execution_batch(batch: WindowBatch, device: str, batch_size: int) -> WindowBatch:
    if len(batch.starts) < batch_size:
        padding = batch_size - len(batch.starts)
        batch = WindowBatch(
            batch.values.cat(batch.values[-1:].expand(padding, *batch.values.shape[1:]), dim=0),
            batch.target.cat(batch.target[-1:].expand(padding, *batch.target.shape[1:]), dim=0),
            batch.observed.cat(
                Tensor.zeros(padding, *batch.observed.shape[1:], dtype=batch.observed.dtype, device=batch.observed.device),
                dim=0,
            ),
            batch.starts + (batch.starts[-1],) * padding,
        )
    values = tuple(value.contiguous().to(device).realize() for value in (batch.values, batch.target, batch.observed))
    return WindowBatch(*values, batch.starts)


def _snapshot(model: Forecast) -> dict[str, Tensor]:
    return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _parameter_count(model: Forecast) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _sparse_calls(model: Forecast, graph: Graph, protocol: Protocol, device: str) -> int:
    output = model(
        Tensor.zeros(1, protocol.train.history, graph.nodes, protocol.features.shape[2], device=device),
        graph,
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
            **settings,
        )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
