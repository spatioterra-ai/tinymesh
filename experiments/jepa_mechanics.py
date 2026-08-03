"""Test joint-embedding prediction over small graph patches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isqrt

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from tinymesh import Graph
from tinymesh.nn import SAGEConv


class PatchEncoder:
    def __init__(self, hidden_features: int) -> None:
        self.conv = SAGEConv(2, hidden_features)
        self.projection = nn.Linear(hidden_features, hidden_features)

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        return self.projection(self.conv(values, graph).relu().mean(axis=-2)).tanh()


class Predictor:
    def __init__(self, hidden_features: int, targets: int) -> None:
        self.hidden = nn.Linear(hidden_features + targets, hidden_features // 2)
        self.output = nn.Linear(hidden_features // 2, hidden_features)

    def __call__(self, context: Tensor, position: Tensor) -> Tensor:
        batch, targets = context.shape[0], position.shape[0]
        context = context.unsqueeze(1).expand(batch, targets, context.shape[1])
        position = position.unsqueeze(0).expand(batch, targets, targets)
        return self.output(self.hidden(context.cat(position, dim=2)).relu())


@dataclass(frozen=True)
class Observation:
    device: str
    seed: int
    steps: int
    samples: int
    hidden_features: int
    learning_rate: float
    ema_decay: float
    initial_loss: float
    aligned_loss: float
    shuffled_target_loss: float
    unconditioned_loss: float
    target_sample_std: float
    target_parameter_delta: float
    target_gradient: float


def compare(
    device: str,
    *,
    seed: int,
    steps: int,
    samples: int,
    hidden_features: int,
    learning_rate: float,
    ema_decay: float,
) -> Observation:
    if steps <= 0 or hidden_features < 2 or learning_rate <= 0 or not 0 <= ema_decay < 1:
        raise ValueError("steps and learning rate must be positive, hidden features >= 2, and EMA decay in [0, 1)")

    Tensor.manual_seed(seed)
    graph = Graph(3, [0, 1, 1, 2], [1, 0, 2, 1])
    context, targets, position = _patches(samples, device)
    online, target = PatchEncoder(hidden_features), PatchEncoder(hidden_features)
    predictor = Predictor(hidden_features, targets.shape[1])
    _update_target(online, target, 0)
    initial_target = {
        name: value.detach().clone().realize()
        for name, value in nn.state.get_state_dict(target).items()
    }
    optimizer = nn.optim.Adam(
        nn.state.get_parameters((online, predictor)),
        lr=learning_rate,
        fused=False,
    )

    def embeddings() -> tuple[Tensor, Tensor]:
        truth = target(targets, graph).detach()
        prediction = predictor(online(context, graph), position)
        return prediction, truth

    initial_loss = _mse(*embeddings()).item()

    @TinyJit
    @Context(TRAINING=1)
    def step(context: Tensor, targets: Tensor, position: Tensor) -> Tensor:
        optimizer.zero_grad()
        truth = target(targets, graph).detach()
        prediction = predictor(online(context, graph), position)
        loss = _mse(prediction, truth).backward()
        return loss.realize(*optimizer.schedule_step())

    for _ in range(steps):
        step(context, targets, position)
        _update_target(online, target, ema_decay)

    prediction, truth = embeddings()
    unconditioned = predictor(online(context, graph), Tensor.zeros_like(position))
    return Observation(
        device=device,
        seed=seed,
        steps=steps,
        samples=samples,
        hidden_features=hidden_features,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
        initial_loss=initial_loss,
        aligned_loss=_mse(prediction, truth).item(),
        shuffled_target_loss=_mse(prediction, truth.flip(0)).item(),
        unconditioned_loss=_mse(unconditioned, truth).item(),
        target_sample_std=truth.std(axis=0).mean().item(),
        target_parameter_delta=sum(
            (value - initial_target[name]).abs().sum().item()
            for name, value in nn.state.get_state_dict(target).items()
        ),
        target_gradient=sum(
            0 if value.grad is None else value.grad.abs().sum().item()
            for value in nn.state.get_parameters(target)
        ),
    )


def _patches(samples: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
    side = isqrt(samples)
    if side < 2 or side * side != samples:
        raise ValueError("samples must be a square number >= 4")

    contexts, targets = [], []
    for index in range(samples):
        a = 4 * (index % side) / (side - 1) - 2
        b = 4 * (index // side) / (side - 1) - 2
        contexts.append([[a, 0], [a, b], [0, b]])
        targets.append([
            [[a + b + 2, a - b + 2], [a + 2, b + 2], [b + 2, a + 2]],
            [[2 * a - b - 2, a + b - 2], [-a - 2, b - 2], [b - 2, -a - 2]],
        ])

    context = Tensor(contexts, device=device).clone().realize()
    target = Tensor(targets, device=device).clone().realize()
    position = Tensor.eye(2).to(device).clone().realize()
    return context, target, position


def _update_target(online: PatchEncoder, target: PatchEncoder, decay: float) -> None:
    source = nn.state.get_state_dict(online)
    for name, value in nn.state.get_state_dict(target).items():
        value.assign(decay * value + (1 - decay) * source[name].detach()).realize()


def _mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def main() -> None:
    observation = compare(
        Device.DEFAULT,
        seed=getenv("SEED", 0),
        steps=getenv("STEPS", 80),
        samples=getenv("SAMPLES", 16),
        hidden_features=getenv("HIDDEN", 8),
        learning_rate=getenv("LR", 0.01),
        ema_decay=getenv("EMA", 0.99),
    )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
