"""Prove edge-vector message learning with GINE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from tinymesh import Graph
from tinymesh.nn import GINEConv


@dataclass(frozen=True)
class Observation:
  device: str
  initial_loss: float
  weight_gradient: tuple[float, float]
  final_loss: float
  reversed_edge_loss: float
  erased_edge_loss: float
  hidden_weight: tuple[float, float]


def fit_one_step(device: str) -> Observation:
  graph = Graph(4, [0, 1], [2, 3])
  values = Tensor([[1.0, 0.0]] * 4, device=device).realize()
  edges = Tensor([[1.0, 0.0], [0.0, 1.0]], device=device).realize()
  target = Tensor([[4.0], [2.0]], device=device).realize()
  model = GINEConv(2, 2, 1)
  model.edge.weight = Tensor.eye(2).to(device).realize()
  model.edge.bias = Tensor.zeros(2, device=device).realize()
  model.hidden.weight = Tensor.zeros(1, 2, device=device).realize()
  model.hidden.bias = Tensor.ones(1, device=device).realize()
  model.output.weight = Tensor.ones(1, 1, device=device).realize()
  model.output.bias = Tensor.zeros(1, device=device).realize()
  optimizer = nn.optim.SGD([model.hidden.weight], lr=0.05, fused=False)

  def loss(edge_values: Tensor) -> Tensor:
    return (model(values, edge_values, graph)[2:] - target).square().mean()

  initial_loss = loss(edges).item()
  with Context(TRAINING=1):
    optimizer.zero_grad()
    training_loss = loss(edges).backward()
    gradient = model.hidden.weight.grad
    if gradient is None:
      raise RuntimeError("GINE parameter has no gradient")
    weight_gradient = tuple(gradient.tolist()[0])
    training_loss.realize(*optimizer.schedule_step())

  return Observation(
    device,
    initial_loss,
    weight_gradient,
    loss(edges).item(),
    loss(edges.flip(0)).item(),
    loss(Tensor.zeros_like(edges)).item(),
    tuple(model.hidden.weight.tolist()[0]),
  )


def main() -> None:
  print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
  main()
