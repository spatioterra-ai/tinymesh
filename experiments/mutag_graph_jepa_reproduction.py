"""Reproduce the official Graph-JEPA MUTAG protocol in tinygrad."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from random import Random
from statistics import fmean

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph
from tinymesh.datasets import MUTAG, mutag


OFFICIAL_REVISION = "72df1b7704921001ea012a21f840300fbc792cdd"
RUN_SEEDS = (42, 21, 95, 12, 35)


@dataclass(frozen=True)
class Fold:
  run: int
  fold: int
  accuracy: float
  final_loss: float
  epochs: int


@dataclass(frozen=True)
class Observation:
  device: str
  official_revision: str
  graphs: int
  patches: int
  hidden_features: int
  epochs: int
  folds: tuple[Fold, ...]
  run_accuracy: tuple[float, ...]
  run_standard_deviation: tuple[float, ...]
  mean_accuracy: float
  standard_deviation: float


class SparseMean:
  def __init__(self, owners: tuple[int, ...], groups: int) -> None:
    items = len(owners)
    self.graph = Graph(items + groups, list(range(items)), [items + owner for owner in owners])
    self.items, self.groups, self.owners = items, groups, owners

  def __call__(self, values: Tensor) -> Tensor:
    zeros = Tensor.zeros(self.groups, values.shape[1], dtype=values.dtype, device=values.device)
    return self.graph.mean(values.cat(zeros, dim=0))[self.items:]


class PaperBatch:
  """Official 32-slot MUTAG patch transform, including empty slots."""

  def __init__(self, data: MUTAG, indices: tuple[int, ...], *, seed: int) -> None:
    source, target, node_labels, edge_labels = [], [], [], []
    node_patch, node_original = [], []
    maximum, mean, masks, adjacency = [], [], [], []
    node_offset = original_offset = 0
    random = Random(seed)
    for row, index in enumerate(indices):
      graph, nodes, edges, _ = data[index]
      node_rows, edge_rows = nodes.tolist(), edges.tolist()
      patches = paper_patches(graph, random)
      rwse = random_walk_encoding(graph, 15)
      masks.append([bool(members) for members in patches])
      adjacency.append([[float(len(set(left) & set(right))) for right in patches] for left in patches])
      maximum.append([
        [max((rwse[node][step] for node in members), default=0.0) for step in range(15)]
        for members in patches
      ])
      mean.append([
        [sum(rwse[node][step] for node in members) / len(members) if members else 0.0 for step in range(15)]
        for members in patches
      ])
      for slot, members in enumerate(patches):
        mapping = {node: node_offset + local for local, node in enumerate(members)}
        for edge, (edge_source, edge_target) in enumerate(zip(graph.source, graph.target)):
          if edge_source in mapping and edge_target in mapping:
            source.append(mapping[edge_source])
            target.append(mapping[edge_target])
            edge_labels.append(edge_rows[edge])
        node_labels.extend(node_rows[node] for node in members)
        node_patch.extend([row * 32 + slot] * len(members))
        node_original.extend([original_offset + node for node in members])
        node_offset += len(members)
      original_offset += graph.nodes

    device = str(data.node_labels[0].device)
    self.nodes = Tensor(node_labels, device=device).one_hot(len(data.node_types)).float().clone().realize()
    self.edges = Tensor(edge_labels, device=device).one_hot(len(data.bond_types)).float().clone().realize()
    self.graph = Graph(node_offset, source, target)
    self.patch_pool = SparseMean(tuple(node_patch), len(indices) * 32)
    self.original_pool = SparseMean(tuple(node_original), original_offset)
    self.node_original = tuple(node_original)
    self.maximum = Tensor(maximum, device=device).clone().realize()
    self.mean = Tensor(mean, device=device).clone().realize()
    self.mask = Tensor(masks, device=device).clone().realize()
    self.adjacency = Tensor(adjacency, device=device).clone().realize()
    self.graphs = len(indices)

  def selection(self, seed: int) -> tuple[Tensor, Tensor]:
    random, context, targets = Random(seed), [], []
    for row in self.mask.tolist():
      present = [slot for slot, value in enumerate(row) if value]
      chosen = random.sample(present, 4)
      context.append(chosen[0])
      targets.append(chosen[1:])
    device = str(self.nodes.device)
    return Tensor(context, device=device).realize(), Tensor(targets, device=device).realize()


class PaperGINE:
  def __init__(self, hidden_features: int) -> None:
    self.eps = Tensor.zeros(1)
    self.hidden = nn.Linear(hidden_features, hidden_features, bias=False)
    self.hidden_norm = nn.BatchNorm(hidden_features)
    self.conv_output = nn.Linear(hidden_features, hidden_features)
    self.norm = nn.BatchNorm(hidden_features)
    self.output = nn.Linear(hidden_features, hidden_features)

  def __call__(self, values: Tensor, edges: Tensor, graph: Graph) -> Tensor:
    message = (graph.edge_values(values, endpoint="source") + edges).relu()
    state = (1 + self.eps) * values + graph.sum_edges(message)
    convolved = self.conv_output(self.hidden_norm(self.hidden(state)).relu())
    return self.output(self.norm(convolved).relu() + values)


class PatchEncoder:
  def __init__(self, node_features: int, edge_features: int, hidden_features: int) -> None:
    self.node = nn.Linear(node_features, hidden_features)
    self.edge = nn.Linear(edge_features, hidden_features)
    self.first = PaperGINE(hidden_features)
    self.update = nn.Linear(hidden_features, hidden_features, bias=False)
    self.update_norm = nn.BatchNorm(hidden_features)
    self.second = PaperGINE(hidden_features)

  def __call__(self, batch: PaperBatch) -> Tensor:
    values, edges = self.node(batch.nodes), self.edge(batch.edges)
    values = self.first(values, edges, batch.graph)
    update = self.update_norm(self.update(batch.patch_pool(values))).relu()
    values = values + update[Tensor(batch.patch_pool.owners, device=str(values.device))]
    original = batch.original_pool(values)
    values = original[Tensor(batch.node_original, device=str(values.device))]
    return batch.patch_pool(self.second(values, edges, batch.graph)).reshape(batch.graphs, 32, -1)


class MLP:
  def __init__(self, in_features: int, out_features: int, layers: int, *, norm: bool, final_activation: bool) -> None:
    self.linears = [nn.Linear(in_features if layer == 0 else in_features, out_features if layer == layers - 1 else in_features,
                              bias=not norm or (layer == layers - 1 and not final_activation)) for layer in range(layers)]
    self.norms = [nn.BatchNorm(out_features if layer == layers - 1 else in_features) for layer in range(layers)] if norm else []
    self.final_activation = final_activation

  def __call__(self, values: Tensor) -> Tensor:
    for layer, linear in enumerate(self.linears):
      values = linear(values)
      if layer < len(self.linears) - 1 or self.final_activation:
        values = self.norms[layer](values) if self.norms else values
        values = values.relu()
    return values


class AttentionLayer:
  def __init__(self, hidden_features: int, heads: int = 8) -> None:
    if hidden_features % heads:
      raise ValueError("hidden features must be divisible by attention heads")
    self.query = nn.Linear(hidden_features, hidden_features)
    self.key = nn.Linear(hidden_features, hidden_features)
    self.value = nn.Linear(hidden_features, hidden_features)
    self.output = nn.Linear(hidden_features, hidden_features)
    self.norm1 = nn.LayerNorm(hidden_features)
    self.norm2 = nn.LayerNorm(hidden_features)
    self.up = nn.Linear(hidden_features, 2 * hidden_features)
    self.down = nn.Linear(2 * hidden_features, hidden_features)
    self.heads, self.width = heads, hidden_features // heads

  def __call__(self, values: Tensor, mask: Tensor | None = None, adjacency: Tensor | None = None) -> Tensor:
    state = self.norm1(values)
    batch, tokens, hidden = state.shape
    query = self.query(state).reshape(batch, tokens, self.heads, self.width).permute(0, 2, 1, 3)
    key = self.key(state).reshape(batch, tokens, self.heads, self.width).permute(0, 2, 1, 3)
    value = self.value(state).reshape(batch, tokens, self.heads, self.width).permute(0, 2, 1, 3)
    score = (query @ key.transpose(2, 3)) / sqrt(self.width)
    if mask is not None:
      score = mask.reshape(batch, 1, 1, tokens).where(float("-inf"), score)
    weight = score.softmax(axis=-1)
    if adjacency is not None:
      weight = weight * adjacency.reshape(batch, 1, tokens, tokens)
    attended = (weight @ value).permute(0, 2, 1, 3).reshape(batch, tokens, hidden)
    values = values + self.output(attended)
    return values + self.down(self.up(self.norm2(values)).relu())


class Encoder:
  def __init__(self, hidden_features: int) -> None:
    self.layers = [AttentionLayer(hidden_features) for _ in range(4)]

  def __call__(self, values: Tensor, mask: Tensor | None = None, adjacency: Tensor | None = None) -> Tensor:
    for layer in self.layers:
      values = layer(values, mask, adjacency)
    return values


class Model:
  def __init__(self, node_features: int, edge_features: int, hidden_features: int) -> None:
    self.patch = PatchEncoder(node_features, edge_features, hidden_features)
    self.position = MLP(15, hidden_features, 1, norm=True, final_activation=True)
    self.context = Encoder(hidden_features)
    self.target = Encoder(hidden_features)
    self.predictor = MLP(hidden_features, 2, 3, norm=False, final_activation=False)

  def predict(self, batch: PaperBatch, context: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    patches = self.patch(batch)
    position = self.position(batch.maximum.reshape(-1, 15)).reshape(batch.graphs, 32, -1)
    rows = Tensor.arange(batch.graphs).to(str(patches.device))
    context_code = (patches[rows, context] + position[rows, context]).unsqueeze(1)
    target_code = patches[rows.unsqueeze(1), targets]
    context_code = self.context(context_code)
    target_adjacency = batch.adjacency[rows.unsqueeze(1).unsqueeze(2), targets.unsqueeze(1), targets.unsqueeze(2)]
    target_code = self.target(target_code.detach(), adjacency=target_adjacency).detach()
    angle = target_code.mean(axis=-1, keepdim=True)
    truth = angle.cosh().cat(angle.sinh(), dim=-1)
    prediction = self.predictor(context_code + position[rows.unsqueeze(1), targets])
    return prediction, truth

  def embed(self, batch: PaperBatch) -> Tensor:
    patches = self.patch(batch)
    position = self.position(batch.mean.reshape(-1, 15)).reshape(batch.graphs, 32, -1)
    encoded = self.target(patches + position, ~batch.mask, batch.adjacency)
    return (encoded * batch.mask.unsqueeze(-1)).sum(axis=1) / batch.mask.sum(axis=1, keepdim=True)


def paper_patches(graph: Graph, random: Random) -> tuple[tuple[int, ...], ...]:
  """Match the official nodes<patches branch and one-hop expansion."""
  slots = list(range(32))
  random.shuffle(slots)
  assigned = slots[:graph.nodes]
  shift = 31 - max(assigned)
  membership = {slot + shift: node for node, slot in enumerate(assigned)}
  neighbors = [set() for _ in range(graph.nodes)]
  for source, target in zip(graph.source, graph.target):
    neighbors[source].add(target)
  return tuple(
    tuple(sorted({node, *neighbors[node]})) if (node := membership.get(slot)) is not None else ()
    for slot in range(32)
  )


def random_walk_encoding(graph: Graph, steps: int) -> tuple[tuple[float, ...], ...]:
  neighbors = [[] for _ in range(graph.nodes)]
  for source, target in zip(graph.source, graph.target):
    neighbors[source].append(target)
  rows = []
  for origin in range(graph.nodes):
    state = {origin: 1.0}
    row = []
    for _ in range(steps):
      following: dict[int, float] = {}
      for source, probability in state.items():
        for target in neighbors[source]:
          following[target] = following.get(target, 0.0) + probability / len(neighbors[source])
      state = following
      row.append(state.get(origin, 0.0))
    rows.append(tuple(row))
  return tuple(rows)


def reproduce(data: MUTAG) -> Observation:
  try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
  except ImportError as error:
    raise RuntimeError("Graph-JEPA reproduction requires numpy and scikit-learn") from error

  labels = np.array(data.labels)
  partitions = tuple(StratifiedKFold(10, shuffle=True, random_state=12345).split(np.zeros(len(data)), labels))
  folds = []
  for run, seed in enumerate(RUN_SEEDS):
    for fold, (train_rows, test_rows) in enumerate(partitions):
      train, test = tuple(map(int, train_rows)), tuple(map(int, test_rows))
      model, test_batches, final_loss, completed = _pretrain(data, train, test, seed * 10 + fold)

      train_embedding = _embed(data, model, train, seed + fold)
      test_embedding = _embed_batches(model, tuple(batch for batch, _ in test_batches))
      classifier = LogisticRegression(max_iter=10_000).fit(train_embedding, labels[list(train)])
      accuracy = float(classifier.score(test_embedding, labels[list(test)]))
      folds.append(Fold(run, fold, accuracy, final_loss, completed))

  result = tuple(folds)
  run_accuracy = tuple(fmean(item.accuracy for item in result if item.run == run) for run in range(5))
  run_deviation = tuple(
    sqrt(fmean((item.accuracy - run_accuracy[run]) ** 2 for item in result if item.run == run))
    for run in range(5)
  )
  return Observation(
    str(data.node_labels[0].device), OFFICIAL_REVISION, len(data), 32, 512, 50, result,
    run_accuracy, run_deviation, fmean(run_accuracy), fmean(run_deviation),
  )


def _pretrain(
  data: MUTAG,
  train: tuple[int, ...],
  test: tuple[int, ...],
  seed: int,
  *,
  hidden_features: int = 512,
  epochs: int = 50,
) -> tuple[Model, tuple[tuple[PaperBatch, tuple[Tensor, Tensor]], ...], float, int]:
  Tensor.manual_seed(seed)
  model = Model(len(data.node_types), len(data.bond_types), hidden_features)
  parameters = nn.state.get_parameters((model.patch, model.position, model.context, model.predictor))
  optimizer = nn.optim.Adam(parameters, lr=0.0005, fused=False)
  random = Random(seed)
  test_batches = tuple(
    (batch, batch.selection(random.randrange(2**31)))
    for offset in range(0, len(test), 128)
    for batch in (PaperBatch(data, test[offset:offset + 128], seed=random.randrange(2**31)),)
  )
  batches_per_epoch = (len(train) + 127) // 128
  best, stale, final_loss, completed = float("inf"), 0, 0.0, 0
  for epoch in range(epochs):
    rows = list(train)
    random.shuffle(rows)
    total_loss = 0.0
    for offset in range(0, len(rows), 128):
      indices = tuple(rows[offset:offset + 128])
      batch = PaperBatch(data, indices, seed=random.randrange(2**31))
      context, targets = batch.selection(random.randrange(2**31))
      with Context(TRAINING=1):
        optimizer.zero_grad()
        prediction, truth = model.predict(batch, context, targets)
        loss = _smooth_l1(prediction, truth).backward()
        loss.realize(*optimizer.schedule_step())
      momentum = 0.996 + epoch * 0.004 / (batches_per_epoch * epochs)
      _update_target(model.context, model.target, momentum)
      total_loss += loss.item() * len(indices)
    final_loss, completed = total_loss / len(train), epoch + 1
    with Context(TRAINING=0):
      test_loss = _evaluation_loss(model, test_batches)
    if test_loss < best * (1 - 1e-4):
      best, stale = test_loss, 0
    else:
      stale += 1
      if stale > 20:
        optimizer.lr.assign(optimizer.lr.item() * 0.5)
        stale = 0
    if optimizer.lr.item() < 1e-5:
      break
  return model, test_batches, final_loss, completed


def _evaluation_loss(model: Model, batches: tuple[tuple[PaperBatch, tuple[Tensor, Tensor]], ...]) -> float:
  losses = []
  for batch, selection in batches:
    prediction, truth = model.predict(batch, *selection)
    # Prevent the complete width-512 forward and scalar loss from becoming one serial Metal kernel.
    prediction.realize()
    truth.realize()
    losses.append(_smooth_l1(prediction, truth).item())
  return fmean(losses)


def _embed(data: MUTAG, model: Model, indices: tuple[int, ...], seed: int) -> list[list[float]]:
  batches = tuple(PaperBatch(data, indices[offset:offset + 128], seed=seed + offset) for offset in range(0, len(indices), 128))
  return _embed_batches(model, batches)


def _embed_batches(model: Model, batches: tuple[PaperBatch, ...]) -> list[list[float]]:
  rows = []
  with Context(TRAINING=0):
    for batch in batches:
      rows.extend(model.embed(batch).tolist())
  return rows


def _smooth_l1(prediction: Tensor, target: Tensor) -> Tensor:
  error = prediction - target
  absolute = error.abs()
  return (absolute < 1).where(0.5 * error.square(), absolute - 0.5).mean()


def _update_target(context: Encoder, target: Encoder, decay: float) -> None:
  for source, value in zip(nn.state.get_parameters(context), nn.state.get_parameters(target)):
    value.assign(decay * value + (1 - decay) * source.detach()).realize()


def main() -> None:
  print(json.dumps(asdict(reproduce(mutag(device=Device.DEFAULT))), indent=2))


if __name__ == "__main__":
  main()
