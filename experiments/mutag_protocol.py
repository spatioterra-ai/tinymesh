"""Shared evaluation policy for MUTAG representation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import fmean

from tinygrad import Context, Tensor, TinyJit, nn

from tinymesh.datasets import MUTAG


@dataclass(frozen=True)
class Metric:
  values: tuple[float, ...]
  mean: float
  standard_deviation: float


@dataclass(frozen=True)
class Probe:
  train_accuracy: float
  test_accuracy: float


def stratified_folds(labels: tuple[int, ...], folds: int, seed: int) -> tuple[tuple[int, ...], ...]:
  counts = [labels.count(label) for label in range(2)]
  if folds < 2 or folds > min(counts):
    raise ValueError(f"folds must be in [2, {min(counts)}]")
  random = Random(seed)
  partitions = [[] for _ in range(folds)]
  for label in range(2):
    indices = [index for index, value in enumerate(labels) if value == label]
    random.shuffle(indices)
    for position, index in enumerate(indices):
      partitions[position % folds].append(index)
  return tuple(tuple(sorted(partition)) for partition in partitions)


def molecular_summary(data: MUTAG) -> Tensor:
  rows = []
  for graph, node_label, edge_label, _ in (data[index] for index in range(len(data))):
    nodes, edges = node_label.tolist(), edge_label.tolist()
    rows.append(
      [nodes.count(label) / len(nodes) for label in range(len(data.node_types))]
      + [edges.count(label) / len(edges) for label in range(len(data.bond_types))]
      + [float(graph.nodes)]
    )
  return Tensor(rows, device=str(data.node_labels[0].device)).clone().realize()


def linear_probe(
  features: Tensor,
  labels: tuple[int, ...],
  train: tuple[int, ...],
  test: tuple[int, ...],
  *,
  steps: int,
  learning_rate: float,
  seed: int,
) -> Probe:
  rows = features.tolist()
  device = str(features.device)
  train_x = Tensor([rows[index] for index in train], device=device).clone().realize()
  test_x = Tensor([rows[index] for index in test], device=device).clone().realize()
  train_y = Tensor([labels[index] for index in train], device=device).clone().realize()
  test_y = Tensor([labels[index] for index in test], device=device).clone().realize()
  mean, std = train_x.mean(axis=0), train_x.std(axis=0)
  scale = (std > 1e-6).where(std, 1)
  train_x = ((train_x - mean) / scale).clone().realize()
  test_x = ((test_x - mean) / scale).clone().realize()

  Tensor.manual_seed(seed)
  model = nn.Linear(train_x.shape[1], 2)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)

  @TinyJit
  @Context(TRAINING=1)
  def step(values: Tensor, target: Tensor) -> Tensor:
    optimizer.zero_grad()
    loss = model(values).sparse_categorical_crossentropy(target).backward()
    return loss.realize(*optimizer.schedule_step())

  for _ in range(steps):
    step(train_x, train_y)
  return Probe(_accuracy(model(train_x), train_y), _accuracy(model(test_x), test_y))


def metric(values: tuple[float, ...]) -> Metric:
  mean = fmean(values)
  return Metric(values, mean, sqrt(fmean((value - mean) ** 2 for value in values)))


def _accuracy(prediction: Tensor, target: Tensor) -> float:
  return (prediction.argmax(axis=1) == target).mean().item()
