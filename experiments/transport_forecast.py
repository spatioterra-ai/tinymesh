"""Test whether recurrent graph models recover known spatial transport."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import gcd, sqrt
from random import Random
from time import perf_counter

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.directed_gru import DiffusionForecast
from tinymesh import Graph
from tinymesh.nn import DirectedDiffusion, GConvGRU


DATA_SEED = 20260729
NODES = 24
TRAIN_TRAJECTORIES = 128
VALIDATION_TRAJECTORIES = 32
TEST_TRAJECTORIES = 32
LOCAL, FORWARD, REVERSE = 0.25, 0.55, 0.20


@dataclass(frozen=True)
class Topology:
  nodes: int
  source: tuple[int, ...]
  target: tuple[int, ...]
  affinity: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class Trajectories:
  values: Tensor

  @property
  def count(self) -> int:
    return int(self.values.shape[0])

  @property
  def steps(self) -> int:
    return int(self.values.shape[1])

  def windows(self, history: int) -> tuple[Tensor, Tensor]:
    if history <= 0 or history >= self.steps:
      raise ValueError(f"history must be in [1, {self.steps})")
    starts = self.steps - history
    values = Tensor.stack(
      *(self.values[:, start : start + history] for start in range(starts)),
      dim=1,
    )
    target = Tensor.stack(
      *(self.values[:, start + history] for start in range(starts)),
      dim=1,
    )
    return (
      values.reshape(self.count * starts, history, *self.values.shape[2:]).realize(),
      target.reshape(self.count * starts, *self.values.shape[2:]).realize(),
    )


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
Operator = Graph | DirectedDiffusion | None


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
  topology = _topology(NODES)
  steps = history + horizon
  train = _trajectories(topology, TRAIN_TRAJECTORIES, steps, DATA_SEED, device)
  validation = _trajectories(
    topology,
    VALIDATION_TRAJECTORIES,
    steps,
    DATA_SEED + 1,
    device,
  )
  test = _trajectories(
    topology,
    TEST_TRAJECTORIES,
    steps,
    DATA_SEED + 2,
    device,
  )

  Tensor.manual_seed(seed)
  model, operator, model_edges = _model(
    model_name,
    topology_name,
    topology,
    hidden_features,
    device,
  )
  best_epoch, runtime, checkpoints = _fit(
    model,
    operator,
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
    model_edges=model_edges,
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
    parameters=_parameter_count(model),
    best_epoch=best_epoch,
    runtime_seconds=runtime,
    checkpoints=checkpoints,
    validation_persistence=_persistence(validation, history, horizon),
    test_persistence=_persistence(test, history, horizon),
    validation=_evaluate(model, operator, validation, history, horizon),
    test=_evaluate(model, operator, test, history, horizon),
  )


def _topology(nodes: int) -> Topology:
  if nodes < 8:
    raise ValueError("nodes must be at least eight")
  edges = [(node, (node + 1) % nodes, 1.0 + 0.05 * (node % 5)) for node in range(nodes)]
  edges.extend(
    (
      node,
      (node + 3 + node % 3) % nodes,
      0.25 + 0.05 * (node % 4),
    )
    for node in range(0, nodes, 2)
  )
  return Topology(
    nodes,
    tuple(source for source, _, _ in edges),
    tuple(target for _, target, _ in edges),
    tuple(affinity for _, _, affinity in edges),
  )


def _permuted(topology: Topology) -> Topology:
  stride = next(value for value in range(topology.nodes - 1, 1, -1) if gcd(value, topology.nodes) == 1)
  permutation = tuple((stride * node + 1) % topology.nodes for node in range(topology.nodes))
  return Topology(
    topology.nodes,
    tuple(permutation[node] for node in topology.source),
    tuple(permutation[node] for node in topology.target),
    topology.affinity,
  )


def _self(topology: Topology) -> Topology:
  nodes = tuple(range(topology.nodes))
  return Topology(topology.nodes, nodes, nodes, (1.0,) * topology.nodes)


def _symmetric(topology: Topology) -> Graph:
  edges = dict.fromkeys(edge for source, target in zip(topology.source, topology.target) for edge in ((source, target), (target, source)))
  return Graph(
    topology.nodes,
    [source for source, _ in edges],
    [target for _, target in edges],
  )


def _trajectories(
  topology: Topology,
  count: int,
  steps: int,
  seed: int,
  device: str,
  *,
  initial: str = "dense",
) -> Trajectories:
  if initial not in ("dense", "pulse"):
    raise ValueError("initial must be 'dense' or 'pulse'")
  random = Random(seed)
  trajectories = []
  for _ in range(count):
    values = _initial(topology.nodes, random, pulse=initial == "pulse")
    mean = sum(values) / topology.nodes
    values = [value - mean for value in values]
    trajectory = [[[value] for value in values]]
    for _ in range(steps - 1):
      values = _step(values, topology)
      trajectory.append([[value] for value in values])
    trajectories.append(trajectory)
  return Trajectories(Tensor(trajectories, device=device).realize())


def _initial(nodes: int, random: Random, *, pulse: bool) -> list[float]:
  if not pulse:
    return [random.uniform(-1, 1) for _ in range(nodes)]
  values = [0.0] * nodes
  selected = random.sample(range(nodes), 4)
  for first, second in zip(selected[::2], selected[1::2]):
    amplitude = random.uniform(0.5, 1)
    values[first], values[second] = amplitude, -amplitude
  return values


def _step(values: list[float], topology: Topology) -> list[float]:
  outgoing, incoming = [0.0] * topology.nodes, [0.0] * topology.nodes
  for source, target, affinity in zip(
    topology.source,
    topology.target,
    topology.affinity,
  ):
    outgoing[source] += affinity
    incoming[target] += affinity

  forward, reverse = [0.0] * topology.nodes, [0.0] * topology.nodes
  for source, target, affinity in zip(
    topology.source,
    topology.target,
    topology.affinity,
  ):
    forward[target] += affinity / outgoing[source] * values[source]
    reverse[source] += affinity / incoming[target] * values[target]
  return [LOCAL * value + FORWARD * downstream + REVERSE * upstream for value, downstream, upstream in zip(values, forward, reverse)]


def _model(
  model_name: str,
  topology_name: str,
  topology: Topology,
  hidden_features: int,
  device: str,
) -> tuple[Model, Operator, int]:
  model: Model
  if model_name == "lstm":
    model = LocalForecast(hidden_features)
  elif model_name == "gconv_gru":
    model = GConvForecast(hidden_features)
  elif model_name == "diffusion_linear":
    model = LinearDiffusionForecast()
  else:
    model = DiffusionForecast(1, hidden_features)
  operator, edges = _operator(model_name, topology_name, topology, device)
  return model, operator, edges


def _operator(
  model_name: str,
  topology_name: str,
  topology: Topology,
  device: str,
) -> tuple[Operator, int]:
  if model_name == "lstm":
    return None, 0
  selected = {
    "true": topology,
    "permuted": _permuted(topology),
    "self": _self(topology),
  }[topology_name]
  if model_name == "gconv_gru":
    graph = _symmetric(selected)
    return graph, graph.edges

  graph = Graph(selected.nodes, selected.source, selected.target)
  affinity = Tensor(selected.affinity, device=device).realize()
  diffusion = DirectedDiffusion(graph, affinity)
  Tensor.realize(diffusion.forward_weight, diffusion.reverse_weight)
  return diffusion, graph.edges


def _fit(
  model: Model,
  operator: Operator,
  train: Trajectories,
  validation: Trajectories,
  *,
  epochs: int,
  history: int,
  batch_size: int,
  learning_rate: float,
) -> tuple[int, float, tuple[Checkpoint, ...]]:
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
      loss = (_predict(model, values, operator) - target).square().mean().backward()
      return loss.realize(*optimizer.schedule_step())

    return step

  start = perf_counter()
  interval = max(1, epochs // 6)
  best_epoch = 0
  best_error = _metrics(_predict(model, validation_values, operator), validation_target).rmse
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
    error = _metrics(
      _predict(model, validation_values, operator, realize_steps=True),
      validation_target,
    ).rmse
    checkpoints.append(Checkpoint(epoch, error))
    if error < best_error:
      best_epoch, best_error, best_state = epoch, error, _snapshot(model)
  runtime = perf_counter() - start
  nn.state.load_state_dict(model, best_state, verbose=False)
  return best_epoch, runtime, tuple(checkpoints)


def _evaluate(
  model: Model,
  operator: Operator,
  data: Trajectories,
  history: int,
  horizon: int,
) -> Evaluation:
  values, target = data.windows(history)
  one_step = _metrics(
    _predict(model, values, operator, realize_steps=True),
    target,
  )
  window = data.values[:, :history]
  predictions = []
  for _ in range(horizon):
    prediction = _predict(model, window, operator, realize_steps=True).realize()
    predictions.append(prediction)
    window = window[:, 1:].cat(prediction.unsqueeze(1), dim=1).realize()
  rollout = Tensor.stack(*predictions, dim=1)
  expected = data.values[:, history : history + horizon]
  return Evaluation(
    one_step,
    _metrics(rollout, expected),
    _metrics(rollout[:, -1], expected[:, -1]),
  )


def _persistence(
  data: Trajectories,
  history: int,
  horizon: int,
) -> Evaluation:
  values, target = data.windows(history)
  one_step = _metrics(values[:, -1], target)
  current = data.values[:, history - 1]
  rollout = current.unsqueeze(1).expand(
    data.count,
    horizon,
    *current.shape[1:],
  )
  expected = data.values[:, history : history + horizon]
  return Evaluation(
    one_step,
    _metrics(rollout, expected),
    _metrics(rollout[:, -1], expected[:, -1]),
  )


def _predict(
  model: Model,
  values: Tensor,
  operator: Operator,
  *,
  realize_steps: bool = False,
) -> Tensor:
  if isinstance(model, LocalForecast):
    return model(values, realize_steps=realize_steps)
  if isinstance(model, GConvForecast):
    if not isinstance(operator, Graph):
      raise ValueError("GConvGRU requires one graph")
    return model(values, operator, realize_steps=realize_steps)
  if not isinstance(operator, DirectedDiffusion):
    raise ValueError("diffusion models require one operator")
  if isinstance(model, LinearDiffusionForecast):
    return model(values, operator)
  return model(values, operator, realize_steps=realize_steps)


def _metrics(prediction: Tensor, target: Tensor) -> Metrics:
  error = prediction - target
  absolute, square = error.abs().mean(), error.square().mean()
  Tensor.realize(absolute, square)
  return Metrics(absolute.item(), sqrt(square.item()))


def _snapshot(model: Model) -> dict[str, Tensor]:
  return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _parameter_count(model: Model) -> int:
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
