"""Prove a trainable GCN caller over CSR aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn

from experiments.csr_aggregation import CSRTopology, csr_edge_sum


class GCN:
    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, values: Tensor, topology: CSRTopology) -> Tensor:
        messages = self.linear(values)
        if not isinstance(messages.device, str):
            raise ValueError("GCN requires one device")
        degree = topology._degree(messages.device)
        scale = (degree != 0).where(degree.maximum(1).cast(messages.dtype).rsqrt(), 0).reshape(-1, 1)
        return csr_edge_sum(messages * scale, topology) * scale


@dataclass(frozen=True)
class Observation:
    device: str
    initial_loss: float
    weight_gradient: float
    final_loss: float
    weight: float


def fit_one_step(device: str) -> Observation:
    topology = CSRTopology(
        4,
        [0, 1, 2, 3, 0, 2, 1, 3],
        [0, 1, 2, 3, 2, 0, 3, 1],
    )
    values = Tensor([[1.0], [-1.0], [0.0], [0.0]], device=device).realize()
    target = Tensor([[1.0], [-1.0]], device=device).realize()
    model = GCN(1, 1)
    model.linear.weight = Tensor.zeros(1, 1, device=device).realize()
    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=2.0, fused=False)

    def loss() -> Tensor:
        return (model(values, topology)[2:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        gradient = model.linear.weight.grad
        if gradient is None:
            raise RuntimeError("GCN parameter has no gradient")
        weight_gradient = gradient.item()
        training_loss.realize(*optimizer.schedule_step())

    return Observation(
        device,
        initial_loss,
        weight_gradient,
        loss().item(),
        model.linear.weight.item(),
    )

def main() -> None:
    print(json.dumps(asdict(fit_one_step(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
