"""Ablate graph-specific JEPA tasks on MUTAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from random import Random

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.mutag_protocol import Metric, Probe, linear_probe, metric, molecular_summary, nearest_label_accuracy, stratified_folds
from tinymesh import Graph
from tinymesh.datasets import MUTAG, mutag
from tinymesh.nn import GINEConv


@dataclass(frozen=True)
class Arm:
  name: str
  position: bool
  hyperbola: bool
  objective: str


ARMS = (
  Arm("euclidean", True, False, "mse"),
  Arm("hyperbola", True, True, "smooth_l1"),
  Arm("hyperbola_mse", True, True, "mse"),
  Arm("positionless", False, True, "smooth_l1"),
)


class SparseMean:
  """Mean item rows into fixed groups through one sparse graph."""

  def __init__(self, owners: tuple[int, ...], groups: int) -> None:
    items = len(owners)
    self.graph = Graph(items + groups, list(range(items)), [items + owner for owner in owners])
    self.items, self.groups, self.owners = items, groups, owners

  def __call__(self, values: Tensor) -> Tensor:
    zeros = Tensor.zeros(self.groups, values.shape[1], dtype=values.dtype, device=values.device)
    return self.graph.mean(values.cat(zeros, dim=0))[self.items:]


class PatchBatch:
  """Sparse disjoint union of fixed random partitions expanded by one hop."""

  def __init__(self, data: MUTAG, indices: tuple[int, ...], *, patches: int, walk_length: int, seed: int) -> None:
    source, target, node_labels, edge_labels = [], [], [], []
    node_patch, patch_graph, positions = [], [], []
    offset = 0
    for row, index in enumerate(indices):
      graph, nodes, edges, _ = data[index]
      node_rows, edge_rows = nodes.tolist(), edges.tolist()
      rwse = _rwse(graph, walk_length)
      for members in _patches(graph, patches, seed + index):
        patch = len(patch_graph)
        mapping = {node: offset + local for local, node in enumerate(members)}
        for edge, (edge_source, edge_target) in enumerate(zip(graph.source, graph.target)):
          if edge_source in mapping and edge_target in mapping:
            source.append(mapping[edge_source])
            target.append(mapping[edge_target])
            edge_labels.append(edge_rows[edge])
        node_labels.extend(node_rows[node] for node in members)
        node_patch.extend([patch] * len(members))
        patch_graph.append(row)
        positions.append([max(rwse[node][step] for node in members) for step in range(walk_length)])
        offset += len(members)

    device = str(data.node_labels[0].device)
    self.nodes = Tensor(node_labels, device=device).one_hot(len(data.node_types)).float().clone().realize()
    self.edges = Tensor(edge_labels, device=device).one_hot(len(data.bond_types)).float().clone().realize()
    self.position = Tensor(positions, device=device).clone().realize()
    self.graph = Graph(offset, source, target)
    self.node_pool = SparseMean(tuple(node_patch), len(patch_graph))
    self.graph_pool = SparseMean(tuple(patch_graph), len(indices))
    self.graphs, self.patches = len(indices), patches

  def selection(self, seed: int, targets: int) -> tuple[Tensor, Tensor]:
    if targets <= 0 or targets >= self.patches:
      raise ValueError(f"targets must be in [1, {self.patches})")
    random, context, target = Random(seed), [], []
    for graph in range(self.graphs):
      order = list(range(self.patches))
      random.shuffle(order)
      context.append(graph * self.patches + order[0])
      target.append([graph * self.patches + patch for patch in order[1:targets + 1]])
    device = str(self.nodes.device)
    return Tensor(context, device=device).realize(), Tensor(target, device=device).realize()


class PatchEncoder:
  def __init__(self, node_features: int, edge_features: int, hidden_features: int) -> None:
    self.first = GINEConv(node_features, edge_features, hidden_features)
    self.second = GINEConv(hidden_features, edge_features, hidden_features)

  def __call__(self, batch: PatchBatch) -> Tensor:
    state = self.first(batch.nodes, batch.edges, batch.graph).relu()
    state = self.second(state, batch.edges, batch.graph).relu()
    return batch.node_pool(state)


class Encoder:
  def __init__(self, hidden_features: int) -> None:
    self.hidden = nn.Linear(hidden_features, hidden_features)
    self.output = nn.Linear(hidden_features, hidden_features)

  def __call__(self, values: Tensor) -> Tensor:
    return self.output(self.hidden(values).relu())


class Predictor:
  def __init__(self, hidden_features: int, out_features: int) -> None:
    self.hidden = nn.Linear(hidden_features, hidden_features // 2)
    self.output = nn.Linear(hidden_features // 2, out_features)

  def __call__(self, context: Tensor, position: Tensor) -> Tensor:
    return self.output(self.hidden(context.unsqueeze(1) + position).relu())


class Model:
  def __init__(self, node_features: int, edge_features: int, walk_length: int, hidden_features: int, arm: Arm) -> None:
    self.patch = PatchEncoder(node_features, edge_features, hidden_features)
    self.online = Encoder(hidden_features)
    self.target = Encoder(hidden_features)
    self.position = nn.Linear(walk_length, hidden_features, bias=False)
    self.predictor = Predictor(hidden_features, 2 if arm.hyperbola else hidden_features)
    self.arm = arm

  def predict(self, batch: PatchBatch, context: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    patch = self.patch(batch)
    position = self.position(batch.position if self.arm.position else Tensor.zeros_like(batch.position))
    context_code = self.online(patch[context] + position[context])
    target_code = self.target(patch[targets].detach()).detach()
    return self.predictor(context_code, position[targets]), _target(target_code, self.arm)

  def embed(self, batch: PatchBatch) -> Tensor:
    patch = self.patch(batch)
    position = self.position(batch.position if self.arm.position else Tensor.zeros_like(batch.position))
    return batch.graph_pool(self.target(patch + position))


@dataclass(frozen=True)
class FoldArm:
  name: str
  parameters: int
  initial_loss: float
  final_loss: float
  initial_target_sample_std: float
  target_sample_std: float
  target_parameter_delta: float
  target_gradient: float
  random_encoder: Probe
  random_retrieval_accuracy: float
  trained_encoder: Probe
  trained_retrieval_accuracy: float


@dataclass(frozen=True)
class FoldResult:
  fold: int
  train_graphs: int
  test_graphs: int
  majority_accuracy: float
  summary: Probe
  summary_retrieval_accuracy: float
  arms: tuple[FoldArm, ...]


@dataclass(frozen=True)
class ArmObservation:
  name: str
  parameters: int
  initial_loss: Metric
  final_loss: Metric
  initial_target_sample_std: Metric
  target_sample_std: Metric
  target_parameter_delta: Metric
  target_gradient: Metric
  random_encoder_accuracy: Metric
  random_retrieval_accuracy: Metric
  trained_encoder_accuracy: Metric
  trained_retrieval_accuracy: Metric
  accuracy_delta: Metric
  retrieval_delta: Metric


@dataclass(frozen=True)
class Observation:
  device: str
  seed: int
  folds: int
  steps: int
  probe_steps: int
  patches: int
  targets: int
  walk_length: int
  hidden_features: int
  learning_rate: float
  ema_decay: float
  probe_learning_rate: float
  majority_accuracy: Metric
  summary_accuracy: Metric
  summary_retrieval_accuracy: Metric
  arms: tuple[ArmObservation, ...]
  results: tuple[FoldResult, ...]


def compare(
  data: MUTAG,
  *,
  seed: int,
  folds: int,
  steps: int,
  probe_steps: int,
  patches: int,
  targets: int,
  walk_length: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
) -> Observation:
  if steps <= 0 or probe_steps <= 0 or hidden_features < 2 or patches < 2 or walk_length <= 0:
    raise ValueError("steps, probe steps, and walk length must be positive; hidden features and patches must be >= 2")
  if learning_rate <= 0 or probe_learning_rate <= 0 or not 0 <= ema_decay < 1:
    raise ValueError("learning rates must be positive and EMA decay must be in [0, 1)")
  if targets <= 0 or targets >= patches:
    raise ValueError("targets must be positive and fewer than patches")

  partitions = stratified_folds(data.labels, folds, seed)
  indices = tuple(range(len(data)))
  summary = molecular_summary(data)
  results = []
  for fold, test in enumerate(partitions):
    test_set = set(test)
    train = tuple(index for index in indices if index not in test_set)
    model_seed = seed * folds + fold
    train_batch = PatchBatch(data, train, patches=patches, walk_length=walk_length, seed=model_seed)
    all_batch = PatchBatch(data, indices, patches=patches, walk_length=walk_length, seed=model_seed)
    majority = max(range(2), key=lambda label: sum(data.labels[index] == label for index in train))
    probe_seed = model_seed + 10_000
    summary_probe = linear_probe(summary, data.labels, train, test, steps=probe_steps, learning_rate=probe_learning_rate, seed=probe_seed)
    results.append(FoldResult(
      fold,
      len(train),
      len(test),
      sum(data.labels[index] == majority for index in test) / len(test),
      summary_probe,
      nearest_label_accuracy(summary, data.labels, train, test),
      tuple(_run_arm(
        data,
        train_batch,
        all_batch,
        train,
        test,
        arm=arm,
        model_seed=model_seed,
        targets=targets,
        steps=steps,
        probe_steps=probe_steps,
        hidden_features=hidden_features,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
        probe_learning_rate=probe_learning_rate,
        probe_seed=probe_seed,
      ) for arm in ARMS),
    ))

  results = tuple(results)
  return Observation(
    str(data.node_labels[0].device),
    seed,
    folds,
    steps,
    probe_steps,
    patches,
    targets,
    walk_length,
    hidden_features,
    learning_rate,
    ema_decay,
    probe_learning_rate,
    metric(tuple(result.majority_accuracy for result in results)),
    metric(tuple(result.summary.test_accuracy for result in results)),
    metric(tuple(result.summary_retrieval_accuracy for result in results)),
    tuple(_aggregate(arm, results) for arm in ARMS),
    results,
  )


def _run_arm(
  data: MUTAG,
  train_batch: PatchBatch,
  all_batch: PatchBatch,
  train: tuple[int, ...],
  test: tuple[int, ...],
  *,
  arm: Arm,
  model_seed: int,
  targets: int,
  steps: int,
  probe_steps: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
  probe_seed: int,
) -> FoldArm:
  Tensor.manual_seed(model_seed)
  model = Model(len(data.node_types), len(data.bond_types), all_batch.position.shape[1], hidden_features, arm)
  _update_target(model.online, model.target, 0)
  initial_target = {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model.target).items()}
  random_embedding = model.embed(all_batch).detach().clone().realize()
  optimizer = nn.optim.Adam(
    nn.state.get_parameters((model.patch, model.online, model.position, model.predictor)),
    lr=learning_rate,
    fused=False,
  )
  evaluation = train_batch.selection(model_seed + 1_000_000, targets)

  def loss(context: Tensor, target: Tensor) -> Tensor:
    prediction, truth = model.predict(train_batch, context, target)
    return _objective(prediction, truth, arm.objective)

  initial_loss = loss(*evaluation).item()

  @TinyJit
  @Context(TRAINING=1)
  def step(context: Tensor, target: Tensor) -> Tensor:
    optimizer.zero_grad()
    value = loss(context, target).backward()
    return value.realize(*optimizer.schedule_step())

  for iteration in range(steps):
    step(*train_batch.selection(model_seed * steps + iteration, targets))
    _update_target(model.online, model.target, ema_decay)

  trained_embedding = model.embed(all_batch).detach().clone().realize()
  random_probe = linear_probe(
    random_embedding,
    data.labels,
    train,
    test,
    steps=probe_steps,
    learning_rate=probe_learning_rate,
    seed=probe_seed,
  )
  trained_probe = linear_probe(
    trained_embedding,
    data.labels,
    train,
    test,
    steps=probe_steps,
    learning_rate=probe_learning_rate,
    seed=probe_seed,
  )
  return FoldArm(
    arm.name,
    sum(parameter.numel() for parameter in nn.state.get_parameters((model.patch, model.online, model.position, model.predictor))),
    initial_loss,
    loss(*evaluation).item(),
    random_embedding.std(axis=0).mean().item(),
    trained_embedding.std(axis=0).mean().item(),
    sum((value - initial_target[name]).abs().sum().item() for name, value in nn.state.get_state_dict(model.target).items()),
    sum(0 if value.grad is None else value.grad.abs().sum().item() for value in nn.state.get_parameters(model.target)),
    random_probe,
    nearest_label_accuracy(random_embedding, data.labels, train, test),
    trained_probe,
    nearest_label_accuracy(trained_embedding, data.labels, train, test),
  )


def _aggregate(arm: Arm, results: tuple[FoldResult, ...]) -> ArmObservation:
  folds = tuple(next(item for item in result.arms if item.name == arm.name) for result in results)
  random_accuracy = tuple(item.random_encoder.test_accuracy for item in folds)
  random_retrieval = tuple(item.random_retrieval_accuracy for item in folds)
  trained_accuracy = tuple(item.trained_encoder.test_accuracy for item in folds)
  trained_retrieval = tuple(item.trained_retrieval_accuracy for item in folds)
  return ArmObservation(
    arm.name,
    folds[0].parameters,
    metric(tuple(item.initial_loss for item in folds)),
    metric(tuple(item.final_loss for item in folds)),
    metric(tuple(item.initial_target_sample_std for item in folds)),
    metric(tuple(item.target_sample_std for item in folds)),
    metric(tuple(item.target_parameter_delta for item in folds)),
    metric(tuple(item.target_gradient for item in folds)),
    metric(random_accuracy),
    metric(random_retrieval),
    metric(trained_accuracy),
    metric(trained_retrieval),
    metric(tuple(trained - random for trained, random in zip(trained_accuracy, random_accuracy))),
    metric(tuple(trained - random for trained, random in zip(trained_retrieval, random_retrieval))),
  )


def _partition(nodes: int, patches: int, seed: int) -> tuple[tuple[int, ...], ...]:
  if patches <= 0 or patches > nodes:
    raise ValueError(f"patches must be in [1, {nodes}]")
  order = list(range(nodes))
  Random(seed).shuffle(order)
  return tuple(tuple(sorted(order[offset::patches])) for offset in range(patches))


def _patches(graph: Graph, patches: int, seed: int) -> tuple[tuple[int, ...], ...]:
  neighbors = [set() for _ in range(graph.nodes)]
  for source, target in zip(graph.source, graph.target):
    neighbors[source].add(target)
    neighbors[target].add(source)
  return tuple(
    tuple(sorted(set(base).union(*(neighbors[node] for node in base))))
    for base in _partition(graph.nodes, patches, seed)
  )


def _rwse(graph: Graph, walk_length: int) -> tuple[tuple[float, ...], ...]:
  if walk_length <= 0:
    raise ValueError("walk length must be positive")
  neighbors = [[] for _ in range(graph.nodes)]
  for source, target in zip(graph.source, graph.target):
    neighbors[source].append(target)
  rows = []
  for origin in range(graph.nodes):
    state = {origin: 1.0}
    row = []
    for _ in range(walk_length):
      following: dict[int, float] = {}
      for source, probability in state.items():
        if neighbors[source]:
          share = probability / len(neighbors[source])
          for target in neighbors[source]:
            following[target] = following.get(target, 0.0) + share
      state = following
      row.append(state.get(origin, 0.0))
    rows.append(tuple(row))
  return tuple(rows)


def _target(values: Tensor, arm: Arm) -> Tensor:
  if not arm.hyperbola:
    return values
  angle = values.mean(axis=-1, keepdim=True)
  return angle.cosh().cat(angle.sinh(), dim=-1)


def _objective(prediction: Tensor, target: Tensor, objective: str) -> Tensor:
  error = prediction - target
  if objective == "mse":
    return error.square().mean()
  if objective == "smooth_l1":
    absolute = error.abs()
    return (absolute < 1).where(0.5 * error.square(), absolute - 0.5).mean()
  raise ValueError(f"unknown objective {objective!r}")


def _update_target(online: Encoder, target: Encoder, decay: float) -> None:
  source = nn.state.get_state_dict(online)
  for name, value in nn.state.get_state_dict(target).items():
    value.assign(decay * value + (1 - decay) * source[name].detach()).realize()


def main() -> None:
  observation = compare(
    mutag(device=Device.DEFAULT),
    seed=getenv("SEED", 0),
    folds=getenv("FOLDS", 5),
    steps=getenv("STEPS", 80),
    probe_steps=getenv("PROBE_STEPS", 150),
    patches=getenv("PATCHES", 8),
    targets=getenv("TARGETS", 3),
    walk_length=getenv("RW", 8),
    hidden_features=getenv("HIDDEN", 16),
    learning_rate=getenv("LR", 0.005),
    ema_decay=getenv("EMA", 0.99),
    probe_learning_rate=getenv("PROBE_LR", 0.05),
  )
  print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
  main()
