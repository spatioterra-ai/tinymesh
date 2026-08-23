"""Test whether recurrent graph models recover known spatial transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.directed_gru import DiffusionForecast
from experiments.transport_protocol import (
  DATA_SEED,
  FORWARD,
  LOCAL,
  NODES,
  REVERSE,
  TEST_TRAJECTORIES,
  TRAIN_TRAJECTORIES,
  VALIDATION_TRAJECTORIES,
  Topology,
  Trajectories,
  permuted,
  self_topology,
  symmetric,
  topology as transport_topology,
  trajectories,
)
from tinymesh import Graph
from tinymesh.nn import DirectedDiffusion, GConvGRU


class LocalForecast:
  def __init__(self, hidden_features: int) -> None:
    self.cell = nn.LSTMCell(1, hidden_features)
    self.readout = nn.Linear(hidden_features, 1)

  def __call__(self, values: Tensor, *, realize_steps: bool = False) -> Tensor:
    batch, _, nodes, features = values.shape
    state = None
    for step in range(values.shape[1]):
      state = self.cell(values[:, step].reshape(batch * nodes, features), state)
      if realize_steps:
        Tensor.realize(*state)
    return self.readout(state[0]).reshape(batch, nodes, 1)


class GConvForecast:
  def __init__(self, hidden_features: int) -> None:
    self.cell = GConvGRU(1, hidden_features, 2)
    self.readout = nn.Linear(hidden_features, 1)

  def __call__(
    self,
    values: Tensor,
    graph: Graph,
    *,
    realize_steps: bool = False,
  ) -> Tensor:
    hidden = None
    for step in range(values.shape[1]):
      hidden = self.cell(values[:, step], graph, hidden)
      if realize_steps:
        hidden.realize()
    return self.readout(hidden)


class LinearDiffusionForecast:
  def __init__(self) -> None:
    self.readout = nn.Linear(3, 1, bias=False)

  def __call__(self, values: Tensor, diffusion: DirectedDiffusion) -> Tensor:
    current = values[:, -1]
    forward, reverse = diffusion(current)
    return self.readout(current.cat(forward, reverse, dim=-1))


Model = LocalForecast | GConvForecast | DiffusionForecast | LinearDiffusionForecast


@dataclass(frozen=True)
class Forecast:
  model: Model
  edges: int
  predict: Callable[[Tensor, bool], Tensor]

  def __call__(self, values: Tensor, *, realize_steps: bool = False) -> Tensor:
    return self.predict(values, realize_steps)


@dataclass(frozen=True)
class Metrics:
  mae: float
  rmse: float


@dataclass(frozen=True)
class Evaluation:
  one_step: Metrics
  rollout: Metrics
  final_horizon: Metrics


@dataclass(frozen=True)
class Checkpoint:
  epoch: int
  validation_rmse: float


@dataclass(frozen=True)
class Observation:
  device: str
  data_seed: int
  seed: int
  model: str
  topology: str
  nodes: int
  source_edges: int
  model_edges: int
  train_trajectories: int
  validation_trajectories: int
  test_trajectories: int
  steps: int
  history: int
  horizon: int
  batch_size: int
  hidden_features: int
  epochs: int
  learning_rate: float
  transport: tuple[float, float, float]
  parameters: int
  best_epoch: int
  runtime_seconds: float
  checkpoints: tuple[Checkpoint, ...]
  validation_persistence: Evaluation
  test_persistence: Evaluation
  validation: Evaluation
  test: Evaluation


def compare(
  device: str,
  *,
  model_name: str = "diffusion_gru",
  topology_name: str = "true",
  seed: int = 0,
  epochs: int = 30,
  history: int = 4,
  horizon: int = 4,
  batch_size: int = 64,
  hidden_features: int = 8,
  learning_rate: float = 0.01,
) -> Observation:
  _validate(
    model_name,
    topology_name,
    seed=seed,
    epochs=epochs,
    history=history,
    horizon=horizon,
    batch_size=batch_size,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
  )
  topology = transport_topology(NODES)
  steps = history + horizon
  train = trajectories(topology, TRAIN_TRAJECTORIES, steps, DATA_SEED, device)
  validation = trajectories(
    topology,
    VALIDATION_TRAJECTORIES,
    steps,
    DATA_SEED + 1,
    device,
  )
  test = trajectories(
    topology,
    TEST_TRAJECTORIES,
    steps,
    DATA_SEED + 2,
    device,
  )

  Tensor.manual_seed(seed)
  forecast = create_forecast(
    model_name,
    topology_name,
    topology,
    hidden_features,
    device,
  )
  best_epoch, runtime, checkpoints = fit(
    forecast,
    train,
    validation,
    epochs=epochs,
    history=history,
    batch_size=batch_size,
    learning_rate=learning_rate,
  )
  return Observation(
    device=device,
    data_seed=DATA_SEED,
    seed=seed,
    model=model_name,
    topology=topology_name,
    nodes=NODES,
    source_edges=len(topology.source),
    model_edges=forecast.edges,
    train_trajectories=TRAIN_TRAJECTORIES,
    validation_trajectories=VALIDATION_TRAJECTORIES,
    test_trajectories=TEST_TRAJECTORIES,
    steps=steps,
    history=history,
    horizon=horizon,
    batch_size=batch_size,
    hidden_features=hidden_features,
    epochs=epochs,
    learning_rate=learning_rate,
    transport=(LOCAL, FORWARD, REVERSE),
    parameters=parameter_count(forecast.model),
    best_epoch=best_epoch,
    runtime_seconds=runtime,
    checkpoints=checkpoints,
    validation_persistence=persistence(validation, history, horizon),
    test_persistence=persistence(test, history, horizon),
    validation=_evaluate(forecast, validation, history, horizon),
    test=_evaluate(forecast, test, history, horizon),
  )


def create_forecast(
  model_name: str,
  topology_name: str,
  topology: Topology,
  hidden_features: int,
  device: str,
) -> Forecast:
  model: Model
  if model_name == "lstm":
    model = LocalForecast(hidden_features)
  elif model_name == "gconv_gru":
    model = GConvForecast(hidden_features)
  elif model_name == "diffusion_linear":
    model = LinearDiffusionForecast()
  else:
    model = DiffusionForecast(1, hidden_features)
  return bind(model, topology_name, topology, device)


def bind(
  model: Model,
  topology_name: str,
  topology: Topology,
  device: str,
) -> Forecast:
  if isinstance(model, LocalForecast):
    return Forecast(model, 0, lambda values, realize_steps: model(values, realize_steps=realize_steps))
  selected = {
    "true": topology,
    "permuted": permuted(topology),
    "self": self_topology(topology),
  }[topology_name]
  if isinstance(model, GConvForecast):
    graph = symmetric(selected)
    return Forecast(model, graph.edges, lambda values, realize_steps: model(values, graph, realize_steps=realize_steps))

  graph = Graph(selected.nodes, selected.source, selected.target)
  affinity = Tensor(selected.affinity, device=device).realize()
  diffusion = DirectedDiffusion(graph, affinity)
  Tensor.realize(diffusion.forward_weight, diffusion.reverse_weight)
  if isinstance(model, LinearDiffusionForecast):
    return Forecast(model, graph.edges, lambda values, _realize_steps: model(values, diffusion))
  return Forecast(model, graph.edges, lambda values, realize_steps: model(values, diffusion, realize_steps=realize_steps))


def fit(
  forecast: Forecast,
  train: Trajectories,
  validation: Trajectories,
  *,
  epochs: int,
  history: int,
  batch_size: int,
  learning_rate: float,
) -> tuple[int, float, tuple[Checkpoint, ...]]:
  model = forecast.model
  optimizer = nn.optim.Adam(
    nn.state.get_parameters(model),
    lr=learning_rate,
    fused=False,
  )
  train_values, train_target = train.windows(history)
  validation_values, validation_target = validation.windows(history)

  def make_step() -> TinyJit:
    @TinyJit
    @Context(TRAINING=1)
    def step(values: Tensor, target: Tensor) -> Tensor:
      optimizer.zero_grad()
      loss = (forecast(values) - target).square().mean().backward()
      return loss.realize(*optimizer.schedule_step())

    return step

  start = perf_counter()
  interval = max(1, epochs // 6)
  best_epoch = 0
  best_error = metrics(forecast(validation_values), validation_target).rmse
  best_state = _snapshot(model)
  checkpoints = [Checkpoint(0, best_error)]
  steps: dict[tuple[int, ...], TinyJit] = {}
  for epoch in range(1, epochs + 1):
    for offset in range(0, train_values.shape[0], batch_size):
      values = train_values[offset : offset + batch_size]
      target = train_target[offset : offset + batch_size]
      step = steps.get(values.shape)
      if step is None:
        steps[values.shape] = step = make_step()
      step(values.contiguous(), target.contiguous())
    if epoch % interval != 0 and epoch != epochs:
      continue
    error = metrics(
      forecast(validation_values, realize_steps=True),
      validation_target,
    ).rmse
    checkpoints.append(Checkpoint(epoch, error))
    if error < best_error:
      best_epoch, best_error, best_state = epoch, error, _snapshot(model)
  runtime = perf_counter() - start
  nn.state.load_state_dict(model, best_state, verbose=False)
  return best_epoch, runtime, tuple(checkpoints)


def _evaluate(
  forecast: Forecast,
  data: Trajectories,
  history: int,
  horizon: int,
) -> Evaluation:
  values, target = data.windows(history)
  one_step = metrics(
    forecast(values, realize_steps=True),
    target,
  )
  window = data.values[:, :history]
  predictions = []
  for _ in range(horizon):
    prediction = forecast(window, realize_steps=True).realize()
    predictions.append(prediction)
    window = window[:, 1:].cat(prediction.unsqueeze(1), dim=1).realize()
  rollout = Tensor.stack(*predictions, dim=1)
  expected = data.values[:, history : history + horizon]
  return Evaluation(
    one_step,
    metrics(rollout, expected),
    metrics(rollout[:, -1], expected[:, -1]),
  )


def persistence(
  data: Trajectories,
  history: int,
  horizon: int,
) -> Evaluation:
  values, target = data.windows(history)
  one_step = metrics(values[:, -1], target)
  current = data.values[:, history - 1]
  rollout = current.unsqueeze(1).expand(
    data.count,
    horizon,
    *current.shape[1:],
  )
  expected = data.values[:, history : history + horizon]
  return Evaluation(
    one_step,
    metrics(rollout, expected),
    metrics(rollout[:, -1], expected[:, -1]),
  )


def metrics(prediction: Tensor, target: Tensor) -> Metrics:
  error = prediction - target
  absolute, square = error.abs().mean(), error.square().mean()
  Tensor.realize(absolute, square)
  return Metrics(absolute.item(), sqrt(square.item()))


def _snapshot(model: Model) -> dict[str, Tensor]:
  return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def parameter_count(model: Model) -> int:
  return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _validate(
  model: str,
  topology: str,
  **settings: int | float,
) -> None:
  combinations = {
    "lstm": {"none"},
    "gconv_gru": {"true", "permuted"},
    "diffusion_gru": {"true", "permuted", "self"},
    "diffusion_linear": {"true", "permuted", "self"},
  }
  if model not in combinations or topology not in combinations[model]:
    choices = ", ".join(f"{name}:{'/'.join(sorted(topologies))}" for name, topologies in combinations.items())
    raise ValueError(f"invalid model/topology; choose from {choices}")
  for name in ("seed", "epochs", "history", "horizon", "batch_size", "hidden_features"):
    value = settings[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= (0 if name != "seed" else -1):
      raise ValueError(f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer")
  if settings["learning_rate"] <= 0:
    raise ValueError("learning_rate must be positive")


def main() -> None:
  observation = compare(
    Device.DEFAULT,
    model_name=getenv("MODEL", "diffusion_gru"),
    topology_name=getenv("TOPOLOGY", "true"),
    seed=getenv("SEED", 0),
    epochs=getenv("EPOCHS", 30),
    history=getenv("HISTORY", 4),
    horizon=getenv("HORIZON", 4),
    batch_size=getenv("BS", 64),
    hidden_features=getenv("HIDDEN", 8),
    learning_rate=getenv("LR", 0.01),
  )
  print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
  main()
