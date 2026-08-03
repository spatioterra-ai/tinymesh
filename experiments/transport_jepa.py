"""Test causal joint-embedding prediction over a node-time product mesh."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.transport_forecast import (
  DATA_SEED,
  NODES,
  TEST_TRAJECTORIES,
  TRAIN_TRAJECTORIES,
  VALIDATION_TRAJECTORIES,
  Topology,
  Trajectories,
  _permuted,
  _topology,
  _trajectories,
)
from tinymesh import Graph


TOPOLOGIES = ("true", "permuted", "temporal", "spatial")


@dataclass(frozen=True, eq=False)
class Mesh:
  graph: Graph
  edge_values: Tensor
  steps: int
  nodes: int
  temporal_edges: int
  spatial_edges: int


class MeshEncoder:
  def __init__(self, hidden_features: int) -> None:
    self.first = MeshConv(1, hidden_features)
    self.middle = MeshConv(hidden_features, hidden_features)
    self.output = MeshConv(hidden_features, hidden_features)
    self.norm = nn.LayerNorm(hidden_features, elementwise_affine=False)
    self.hidden_features = hidden_features

  def __call__(self, values: Tensor, mesh: Mesh) -> Tensor:
    expected = (mesh.steps, mesh.nodes, 1)
    if values.ndim != 4 or values.shape[1:] != expected:
      raise ValueError(f"values must have shape [B, {mesh.steps}, {mesh.nodes}, 1], got {values.shape}")
    state = values.reshape(values.shape[0], mesh.graph.nodes, 1)
    state = self.first(state, mesh).tanh()
    state = self.middle(state, mesh).tanh()
    state = self.norm(self.output(state, mesh))
    return state.reshape(values.shape[0], mesh.steps, mesh.nodes, self.hidden_features)


class MeshConv:
  """Relation-weighted messages with no topology-only signal."""

  def __init__(self, in_features: int, out_features: int) -> None:
    self.root = nn.Linear(in_features, out_features, bias=False)
    self.relation = nn.Linear(in_features, 3 * out_features, bias=False)
    self.out_features = out_features

  def __call__(self, values: Tensor, mesh: Mesh) -> Tensor:
    source = mesh.graph.edge_values(values, endpoint="source")
    relation = self.relation(source).reshape(*source.shape[:-1], 3, self.out_features)
    edge = mesh.edge_values.reshape((1,) * (values.ndim - 2) + (*mesh.edge_values.shape, 1))
    message = (relation * edge).sum(axis=-2)
    return self.root(values) + mesh.graph.sum_edges(message)


class Predictor:
  def __init__(self, hidden_features: int) -> None:
    self.hidden = nn.Linear(hidden_features, hidden_features)
    self.output = nn.Linear(hidden_features, hidden_features)

  def __call__(self, context: Tensor) -> Tensor:
    return self.output(self.hidden(context).relu())


class Model:
  def __init__(self, hidden_features: int) -> None:
    self.online = MeshEncoder(hidden_features)
    self.target = MeshEncoder(hidden_features)
    self.predictor = Predictor(hidden_features)

  def loss(self, context: Tensor, target: Tensor, context_mesh: Mesh, target_mesh: Mesh) -> Tensor:
    prediction = self.predictor(self.online(context, context_mesh))
    truth = self.target(target, target_mesh).detach()
    return (prediction - truth).abs().mean()


@dataclass(frozen=True)
class Metrics:
  mae: float
  rmse: float


@dataclass(frozen=True)
class Probe:
  validation: Metrics
  test: Metrics


@dataclass(frozen=True)
class ArmObservation:
  name: str
  product_nodes: int
  product_edges: int
  temporal_edges: int
  spatial_edges: int
  parameters: int
  initial_loss: float
  final_loss: float
  initial_embedding_std: float
  trained_embedding_std: float
  target_parameter_delta: float
  target_gradient: float
  random_encoder: Probe
  trained_encoder: Probe
  validation_rmse_gain: float
  test_rmse_gain: float


@dataclass(frozen=True)
class Observation:
  device: str
  data_seed: int
  seed: int
  nodes: int
  source_edges: int
  train_trajectories: int
  validation_trajectories: int
  test_trajectories: int
  history: int
  horizon: int
  steps: int
  probe_steps: int
  hidden_features: int
  learning_rate: float
  ema_decay: float
  probe_learning_rate: float
  persistence: Probe
  raw_history: Probe
  arms: tuple[ArmObservation, ...]


def compare(
  device: str,
  *,
  seed: int,
  history: int,
  horizon: int,
  steps: int,
  probe_steps: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
) -> Observation:
  _validate(
    seed=seed,
    history=history,
    horizon=horizon,
    steps=steps,
    probe_steps=probe_steps,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
    ema_decay=ema_decay,
    probe_learning_rate=probe_learning_rate,
  )
  topology = _topology(NODES)
  total_steps = history + horizon
  train = _trajectories(topology, TRAIN_TRAJECTORIES, total_steps, DATA_SEED, device)
  validation = _trajectories(topology, VALIDATION_TRAJECTORIES, total_steps, DATA_SEED + 1, device)
  test = _trajectories(topology, TEST_TRAJECTORIES, total_steps, DATA_SEED + 2, device)
  train_context, train_target = _blocks(train, history)
  validation_context, validation_target = _blocks(validation, history)
  test_context, test_target = _blocks(test, history)
  raw_train = _raw_history(train_context)
  raw_validation = _raw_history(validation_context)
  raw_test = _raw_history(test_context)

  return Observation(
    device,
    DATA_SEED,
    seed,
    topology.nodes,
    len(topology.source),
    train.count,
    validation.count,
    test.count,
    history,
    horizon,
    steps,
    probe_steps,
    hidden_features,
    learning_rate,
    ema_decay,
    probe_learning_rate,
    _persistence(validation_context, validation_target, test_context, test_target),
    _probe(
      raw_train,
      train_target,
      raw_validation,
      validation_target,
      raw_test,
      test_target,
      steps=probe_steps,
      learning_rate=probe_learning_rate,
      seed=seed + 20_000,
    ),
    tuple(_run_arm(
      topology,
      name,
      train_context,
      train_target,
      validation_context,
      validation_target,
      test_context,
      test_target,
      seed=seed,
      steps=steps,
      probe_steps=probe_steps,
      hidden_features=hidden_features,
      learning_rate=learning_rate,
      ema_decay=ema_decay,
      probe_learning_rate=probe_learning_rate,
    ) for name in TOPOLOGIES),
  )


def _run_arm(
  topology: Topology,
  name: str,
  train_context: Tensor,
  train_target: Tensor,
  validation_context: Tensor,
  validation_target: Tensor,
  test_context: Tensor,
  test_target: Tensor,
  *,
  seed: int,
  steps: int,
  probe_steps: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
) -> ArmObservation:
  device = str(train_context.device)
  context_mesh = _mesh(topology, name, train_context.shape[1], device)
  target_mesh = _mesh(topology, name, train_target.shape[1], device)
  Tensor.manual_seed(seed)
  model = Model(hidden_features)
  _update_target(model.online, model.target, 0)
  initial_target = _snapshot(model.target)
  random_train = _embedding(model.target, train_context, context_mesh)
  random_validation = _embedding(model.target, validation_context, context_mesh)
  random_test = _embedding(model.target, test_context, context_mesh)
  initial_std = _variation(random_train)
  initial_loss = model.loss(train_context, train_target, context_mesh, target_mesh).item()
  optimizer = nn.optim.Adam(
    nn.state.get_parameters((model.online, model.predictor)),
    lr=learning_rate,
    fused=False,
  )

  @TinyJit
  @Context(TRAINING=1)
  def step(context: Tensor, target: Tensor) -> Tensor:
    optimizer.zero_grad()
    loss = model.loss(context, target, context_mesh, target_mesh).backward()
    return loss.realize(*optimizer.schedule_step())

  for _ in range(steps):
    step(train_context, train_target)
    _update_target(model.online, model.target, ema_decay)

  trained_train = _embedding(model.target, train_context, context_mesh)
  trained_validation = _embedding(model.target, validation_context, context_mesh)
  trained_test = _embedding(model.target, test_context, context_mesh)
  probe_seed = seed + 10_000
  random_probe = _probe(
    random_train,
    train_target,
    random_validation,
    validation_target,
    random_test,
    test_target,
    steps=probe_steps,
    learning_rate=probe_learning_rate,
    seed=probe_seed,
  )
  trained_probe = _probe(
    trained_train,
    train_target,
    trained_validation,
    validation_target,
    trained_test,
    test_target,
    steps=probe_steps,
    learning_rate=probe_learning_rate,
    seed=probe_seed,
  )
  return ArmObservation(
    name,
    context_mesh.graph.nodes,
    context_mesh.graph.edges,
    context_mesh.temporal_edges,
    context_mesh.spatial_edges,
    sum(parameter.numel() for parameter in nn.state.get_parameters((model.online, model.predictor))),
    initial_loss,
    model.loss(train_context, train_target, context_mesh, target_mesh).item(),
    initial_std,
    _variation(trained_train),
    sum((value - initial_target[key]).abs().sum().item() for key, value in nn.state.get_state_dict(model.target).items()),
    sum(0 if value.grad is None else value.grad.abs().sum().item() for value in nn.state.get_parameters(model.target)),
    random_probe,
    trained_probe,
    random_probe.validation.rmse - trained_probe.validation.rmse,
    random_probe.test.rmse - trained_probe.test.rmse,
  )


def _mesh(topology: Topology, name: str, steps: int, device: str) -> Mesh:
  if name not in TOPOLOGIES:
    raise ValueError(f"unknown topology {name!r}")
  if steps <= 0:
    raise ValueError("mesh steps must be positive")
  selected = _permuted(topology) if name == "permuted" else topology
  time = Graph(steps, list(range(steps - 1)), list(range(1, steps))) if name != "spatial" else Graph(steps, [], [])
  if name == "temporal":
    space, spatial_values = Graph(topology.nodes, [], []), []
  else:
    forward, reverse = _normalized(selected)
    space = Graph(selected.nodes, [*selected.source, *selected.target], [*selected.target, *selected.source])
    spatial_values = [[0.0, weight, 0.0] for weight in forward] + [[0.0, 0.0, weight] for weight in reverse]
  graph = time.cartesian(space)
  temporal_edges = time.edges * space.nodes
  edge_values = [[1.0, 0.0, 0.0]] * temporal_edges + spatial_values * time.nodes
  return Mesh(
    graph,
    Tensor(edge_values, device=device).realize(),
    steps,
    topology.nodes,
    temporal_edges,
    time.nodes * space.edges,
  )


def _normalized(topology: Topology) -> tuple[list[float], list[float]]:
  outgoing, incoming = [0.0] * topology.nodes, [0.0] * topology.nodes
  for source, target, affinity in zip(topology.source, topology.target, topology.affinity):
    outgoing[source] += affinity
    incoming[target] += affinity
  return (
    [affinity / outgoing[source] for source, affinity in zip(topology.source, topology.affinity)],
    [affinity / incoming[target] for target, affinity in zip(topology.target, topology.affinity)],
  )


def _blocks(data: Trajectories, history: int) -> tuple[Tensor, Tensor]:
  return data.values[:, :history].contiguous().realize(), data.values[:, history:].contiguous().realize()


def _raw_history(context: Tensor) -> Tensor:
  return context[..., 0].permute(0, 2, 1).contiguous().realize()


def _future(target: Tensor) -> Tensor:
  return target[..., 0].permute(0, 2, 1).contiguous().realize()


def _embedding(encoder: MeshEncoder, values: Tensor, mesh: Mesh) -> Tensor:
  state = encoder(values, mesh).detach()
  return state.permute(0, 2, 1, 3).reshape(values.shape[0], mesh.nodes, -1).contiguous().realize()


def _variation(values: Tensor) -> float:
  return values.reshape(-1, values.shape[-1]).std(axis=0).mean().item()


def _probe(
  train: Tensor,
  train_target: Tensor,
  validation: Tensor,
  validation_target: Tensor,
  test: Tensor,
  test_target: Tensor,
  *,
  steps: int,
  learning_rate: float,
  seed: int,
) -> Probe:
  train, validation, test = _standardize(train, validation, test)
  Tensor.manual_seed(seed)
  model = nn.Linear(train.shape[-1], train_target.shape[1])
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
  expected = _future(train_target)

  @TinyJit
  @Context(TRAINING=1)
  def step(values: Tensor, target: Tensor) -> Tensor:
    optimizer.zero_grad()
    loss = (model(values) - target).square().mean().backward()
    return loss.realize(*optimizer.schedule_step())

  for _ in range(steps):
    step(train, expected)
  return Probe(
    _metrics(model(validation), _future(validation_target)),
    _metrics(model(test), _future(test_target)),
  )


def _standardize(train: Tensor, validation: Tensor, test: Tensor) -> tuple[Tensor, Tensor, Tensor]:
  width = train.shape[-1]
  rows = train.reshape(-1, width)
  mean = rows.mean(axis=0).reshape(1, 1, width)
  scale = rows.std(axis=0).maximum(1e-6).reshape(1, 1, width)
  return (
    ((train - mean) / scale).contiguous().realize(),
    ((validation - mean) / scale).contiguous().realize(),
    ((test - mean) / scale).contiguous().realize(),
  )


def _persistence(
  validation_context: Tensor,
  validation_target: Tensor,
  test_context: Tensor,
  test_target: Tensor,
) -> Probe:
  def predict(context: Tensor, target: Tensor) -> Tensor:
    return context[:, -1, :, 0].unsqueeze(-1).expand(context.shape[0], context.shape[2], target.shape[1])

  return Probe(
    _metrics(predict(validation_context, validation_target), _future(validation_target)),
    _metrics(predict(test_context, test_target), _future(test_target)),
  )


def _metrics(prediction: Tensor, target: Tensor) -> Metrics:
  error = prediction - target
  absolute, square = error.abs().mean(), error.square().mean()
  Tensor.realize(absolute, square)
  return Metrics(absolute.item(), sqrt(square.item()))


def _snapshot(model: object) -> dict[str, Tensor]:
  return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _update_target(online: MeshEncoder, target: MeshEncoder, decay: float) -> None:
  source = nn.state.get_state_dict(online)
  Tensor.realize(*source.values())
  for name, value in nn.state.get_state_dict(target).items():
    value.assign(decay * value + (1 - decay) * source[name].detach()).realize()


def _validate(**settings: int | float) -> None:
  for name in ("history", "horizon", "steps", "probe_steps"):
    if not isinstance(settings[name], int) or isinstance(settings[name], bool) or settings[name] <= 0:
      raise ValueError(f"{name} must be a positive integer")
  if not isinstance(settings["seed"], int) or isinstance(settings["seed"], bool) or settings["seed"] < 0:
    raise ValueError("seed must be a non-negative integer")
  if not isinstance(settings["hidden_features"], int) or isinstance(settings["hidden_features"], bool) or settings["hidden_features"] < 2:
    raise ValueError("hidden features must be an integer >= 2")
  if settings["learning_rate"] <= 0 or settings["probe_learning_rate"] <= 0:
    raise ValueError("learning rates must be positive")
  if not 0 <= settings["ema_decay"] < 1:
    raise ValueError("EMA decay must be in [0, 1)")
  if settings["history"] != settings["horizon"]:
    raise ValueError("history and horizon must match for tokenwise prediction")


def main() -> None:
  observation = compare(
    Device.DEFAULT,
    seed=getenv("SEED", 0),
    history=getenv("HISTORY", 4),
    horizon=getenv("HORIZON", 4),
    steps=getenv("STEPS", 100),
    probe_steps=getenv("PROBE_STEPS", 150),
    hidden_features=getenv("HIDDEN", 8),
    learning_rate=getenv("LR", 0.001),
    ema_decay=getenv("EMA", 0.998),
    probe_learning_rate=getenv("PROBE_LR", 0.05),
  )
  print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
  main()
