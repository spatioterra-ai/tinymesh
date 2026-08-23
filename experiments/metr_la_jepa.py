"""Test causal factorized JEPA representations on METR-LA."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from itertools import islice
from math import prod
from random import Random

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops

from experiments.metr_la_protocol import Protocol, Scores, WindowBatch, WindowSpan, batches, execution_batch, operators, prepare, score
from tinymesh.datasets import metr_la
from tinymesh.nn import DirectedDiffusion


ARMS = ("factorized", "permuted", "temporal", "spatial")
DATA_SEED = 20_260_803


class SpatialEncoder:
  """Add a learned bidirectional residual to a node-local encoding."""

  def __init__(self, in_features: int, hidden_features: int) -> None:
    self.root = nn.Linear(in_features, hidden_features, bias=False)
    self.neighbor = nn.Linear(2 * in_features, hidden_features, bias=False)
    self.gate = Tensor.zeros(hidden_features)
    self.norm = nn.LayerNorm(hidden_features, elementwise_affine=False)
    self.in_features, self.hidden_features = in_features, hidden_features

  def __call__(self, values: Tensor, diffusion: DirectedDiffusion) -> Tensor:
    expected = (diffusion.graph.nodes, self.in_features)
    if values.ndim < 3 or values.shape[-2:] != expected:
      raise ValueError(f"values must have shape [..., T, {diffusion.graph.nodes}, {self.in_features}], got {values.shape}")
    forward, reverse = diffusion(values)
    transport = (forward - values).cat(reverse - values, dim=-1)
    state = self.root(values) + self.neighbor(transport) * self.gate.tanh()
    return self.norm(state.tanh())


class FactorizedEncoder:
  """Apply directed spatial encoding before node-local causal time mixing."""

  def __init__(self, in_features: int, hidden_features: int, periods: int) -> None:
    self.spatial = SpatialEncoder(in_features, hidden_features)
    self.temporal = nn.LSTMCell(hidden_features, hidden_features, bias=False)
    self.norm = nn.LayerNorm(hidden_features, elementwise_affine=False)
    self.hidden_features, self.periods = hidden_features, periods

  def __call__(self, values: Tensor, diffusion: DirectedDiffusion) -> Tensor:
    spatial = self.spatial(values, diffusion)
    periods = spatial.shape[-3]
    if periods > self.periods:
      raise ValueError(f"values must contain at most {self.periods} periods, got {periods}")
    leading, nodes = spatial.shape[:-3], spatial.shape[-2]
    rows = prod(leading) * nodes
    zero = Tensor.zeros(rows, self.hidden_features, dtype=spatial.dtype, device=spatial.device)
    state, states = (zero, zero), []
    for period in range(periods):
      state = self.temporal(spatial[..., period, :, :].reshape(rows, self.hidden_features), state)
      states.append(state[0].reshape(*leading, nodes, self.hidden_features))
    return self.norm(Tensor.stack(*states, dim=-3))


Encoder = SpatialEncoder | FactorizedEncoder
Representation = Callable[[Tensor], Tensor]


class Predictor:
  def __init__(self, in_features: int, out_features: int) -> None:
    self.hidden = nn.Linear(in_features, in_features)
    self.output = nn.Linear(in_features, out_features)

  def __call__(self, values: Tensor) -> Tensor:
    return self.output(self.hidden(values).relu())


class Model:
  def __init__(self, in_features: int, hidden_features: int, history: int, horizon: int, *, temporal: bool) -> None:
    periods = max(history, horizon)
    self.online = FactorizedEncoder(in_features, hidden_features, periods) if temporal else SpatialEncoder(in_features, hidden_features)
    self.target = FactorizedEncoder(in_features, hidden_features, periods) if temporal else SpatialEncoder(in_features, hidden_features)
    self.predictor = Predictor(history * hidden_features, horizon * hidden_features)

  def loss(self, context: Tensor, target: Tensor, diffusion: DirectedDiffusion) -> Tensor:
    prediction = self.predictor(_embedding(self.online, diffusion, context))
    truth = _embedding(self.target, diffusion, target).detach()
    return (prediction - truth).abs().mean()


@dataclass(frozen=True)
class Probe:
  validation: Scores
  test: Scores | None


@dataclass(frozen=True)
class ArmObservation:
  name: str
  topology: str
  temporal: bool
  parameters: int
  sparse_calls: int
  initial_loss: float
  final_loss: float
  initial_embedding_std: float
  trained_embedding_std: float
  spatial_gate: float
  target_parameter_delta: float
  target_gradient: float
  random_encoder: Probe
  trained_encoder: Probe
  validation_rmse_gain: float
  test_rmse_gain: float | None


@dataclass(frozen=True)
class Observation:
  device: str
  seed: int
  nodes: int
  edges: int
  train_windows: int
  validation_windows: int
  test_windows: int
  history: int
  horizon: int
  steps: int
  probe_steps: int
  probe_samples: int
  evaluation_samples: int
  batch_size: int
  hidden_features: int
  learning_rate: float
  ema_decay: float
  probe_learning_rate: float
  evaluate_test: bool
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
  probe_samples: int,
  evaluation_samples: int,
  batch_size: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
  evaluate_test: bool,
) -> Observation:
  return evaluate(
    prepare(metr_la(device="CPU"), history=history, horizon=horizon, feature_set="calendar"),
    device=device,
    seed=seed,
    steps=steps,
    probe_steps=probe_steps,
    probe_samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
    ema_decay=ema_decay,
    probe_learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  )


def evaluate(
  protocol: Protocol,
  *,
  device: str,
  seed: int,
  steps: int,
  probe_steps: int,
  probe_samples: int,
  evaluation_samples: int,
  batch_size: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
  evaluate_test: bool = False,
) -> Observation:
  _validate(
    protocol,
    seed=seed,
    steps=steps,
    probe_steps=probe_steps,
    probe_samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
    ema_decay=ema_decay,
    probe_learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  )
  tensors = tuple(value.to(device).realize() for value in (
    protocol.features[..., :2].contiguous(),
    protocol.target,
    protocol.observed,
  ))
  selected_operators = operators(protocol, "local_diffusion", device)
  persistence = Probe(
    _persistence(protocol, protocol.validation, tensors, evaluation_samples, batch_size, DATA_SEED + 1),
    _persistence(protocol, protocol.test, tensors, evaluation_samples, batch_size, DATA_SEED + 2) if evaluate_test else None,
  )
  raw_history, _ = _probe(
    _raw_history,
    protocol,
    tensors,
    device=device,
    seed=seed + 20_000,
    steps=probe_steps,
    samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  )
  arms = tuple(_run_arm(
    name,
    selected_operators,
    protocol,
    tensors,
    device=device,
    seed=seed,
    steps=steps,
    probe_steps=probe_steps,
    probe_samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
    ema_decay=ema_decay,
    probe_learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  ) for name in ARMS)
  return Observation(
    device,
    seed,
    protocol.data.graph.nodes,
    protocol.data.graph.edges,
    protocol.train.windows,
    protocol.validation.windows,
    protocol.test.windows,
    protocol.train.history,
    protocol.train.horizon,
    steps,
    probe_steps,
    probe_samples,
    evaluation_samples,
    batch_size,
    hidden_features,
    learning_rate,
    ema_decay,
    probe_learning_rate,
    evaluate_test,
    persistence,
    raw_history,
    arms,
  )


def _run_arm(
  name: str,
  operators: dict[str, DirectedDiffusion],
  protocol: Protocol,
  tensors: tuple[Tensor, Tensor, Tensor],
  *,
  device: str,
  seed: int,
  steps: int,
  probe_steps: int,
  probe_samples: int,
  evaluation_samples: int,
  batch_size: int,
  hidden_features: int,
  learning_rate: float,
  ema_decay: float,
  probe_learning_rate: float,
  evaluate_test: bool,
) -> ArmObservation:
  topology, temporal, diffusion = _arm(name, operators)
  Tensor.manual_seed(seed)
  model = Model(2, hidden_features, protocol.train.history, protocol.train.horizon, temporal=temporal)
  _update_target(model.online, model.target, 0)
  initial_target = _snapshot(model.target)
  fixed = execution_batch(
    next(batches(protocol, protocol.train, batch_size, shuffle=DATA_SEED, tensors=tensors)),
    device,
    batch_size,
  )
  context, target = _blocks(fixed)
  initial_loss = model.loss(context, target, diffusion).item()
  represent = partial(_embedding, model.target, diffusion)
  random_probe, initial_std = _probe(
    represent,
    protocol,
    tensors,
    device=device,
    seed=seed + 10_000,
    steps=probe_steps,
    samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  )
  optimizer = nn.optim.Adam(nn.state.get_parameters((model.online, model.predictor)), lr=learning_rate, fused=False)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(values: Tensor, future: Tensor) -> Tensor:
    optimizer.zero_grad()
    loss = model.loss(values, future, diffusion).backward()
    return loss.realize(*optimizer.schedule_step())

  for step in range(steps):
    batch = execution_batch(
      next(batches(protocol, protocol.train, batch_size, shuffle=DATA_SEED + step, tensors=tensors)),
      device,
      batch_size,
    )
    train_step(*_blocks(batch))
    _update_target(model.online, model.target, ema_decay)

  trained_probe, trained_std = _probe(
    represent,
    protocol,
    tensors,
    device=device,
    seed=seed + 10_000,
    steps=probe_steps,
    samples=probe_samples,
    evaluation_samples=evaluation_samples,
    batch_size=batch_size,
    learning_rate=probe_learning_rate,
    evaluate_test=evaluate_test,
  )
  random_test = random_probe.test.overall.rmse if random_probe.test is not None else None
  trained_test = trained_probe.test.overall.rmse if trained_probe.test is not None else None
  return ArmObservation(
    name,
    topology,
    temporal,
    sum(parameter.numel() for parameter in nn.state.get_parameters((model.online, model.predictor))),
    _sparse_calls(model, context, target, diffusion),
    initial_loss,
    model.loss(context, target, diffusion).item(),
    initial_std,
    trained_std,
    _spatial(model.target).gate.tanh().abs().mean().item(),
    sum((value - initial_target[key]).abs().sum().item() for key, value in nn.state.get_state_dict(model.target).items()),
    sum(0 if value.grad is None else value.grad.abs().sum().item() for value in nn.state.get_parameters(model.target)),
    random_probe,
    trained_probe,
    random_probe.validation.overall.rmse - trained_probe.validation.overall.rmse,
    None if random_test is None or trained_test is None else random_test - trained_test,
  )


def _probe(
  represent: Representation,
  protocol: Protocol,
  tensors: tuple[Tensor, Tensor, Tensor],
  *,
  device: str,
  seed: int,
  steps: int,
  samples: int,
  evaluation_samples: int,
  batch_size: int,
  learning_rate: float,
  evaluate_test: bool,
) -> tuple[Probe, float]:
  values, target, observed = _sample(
    represent,
    protocol,
    tensors,
    device=device,
    samples=samples,
    batch_size=batch_size,
  )
  width = values.shape[-1]
  variation = _variation(values)
  rows = values.reshape(-1, width)
  mean = rows.mean(axis=0).reshape(1, 1, width)
  scale = rows.std(axis=0).maximum(1e-6).reshape(1, 1, width)
  values = ((values - mean) / scale).contiguous().realize()
  Tensor.manual_seed(seed)
  model = nn.Linear(width, protocol.train.horizon)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(features: Tensor, expected: Tensor, mask: Tensor) -> Tensor:
    optimizer.zero_grad()
    error = model(features) - expected
    weight = mask.cast(error.dtype)
    loss = (error.square() * weight).sum() / weight.sum()
    return loss.backward().realize(*optimizer.schedule_step())

  random = Random(seed)
  for _ in range(steps):
    index = Tensor(random.sample(range(samples), batch_size), device=device)
    train_step(values[index], target[index], observed[index])
  validation = _probe_scores(
    model, represent, protocol, protocol.validation, tensors, mean, scale, device,
    evaluation_samples, batch_size, DATA_SEED + 1,
  )
  test = (
    _probe_scores(
      model, represent, protocol, protocol.test, tensors, mean, scale, device,
      evaluation_samples, batch_size, DATA_SEED + 2,
    )
    if evaluate_test
    else None
  )
  return Probe(validation, test), variation


def _sample(
  represent: Representation,
  protocol: Protocol,
  tensors: tuple[Tensor, Tensor, Tensor],
  *,
  device: str,
  samples: int,
  batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
  values, targets, masks = [], [], []
  for batch in _sampled_batches(protocol, protocol.train, tensors, samples, batch_size, DATA_SEED):
    batch = execution_batch(batch, device, batch_size)
    values.append(represent(batch.values).detach().realize())
    targets.append(batch.target)
    masks.append(batch.observed)
  return Tensor.cat(*values, dim=0).realize(), Tensor.cat(*targets, dim=0).realize(), Tensor.cat(*masks, dim=0).realize()


def _probe_scores(
  model: nn.Linear,
  represent: Representation,
  protocol: Protocol,
  span: WindowSpan,
  tensors: tuple[Tensor, Tensor, Tensor],
  mean: Tensor,
  scale: Tensor,
  device: str,
  samples: int,
  batch_size: int,
  shuffle: int,
) -> Scores:
  predict = _probe_predictor(model, represent, mean, scale)

  def errors():
    for batch in _sampled_batches(protocol, span, tensors, samples, batch_size, shuffle):
      size = len(batch.starts)
      execution = execution_batch(batch, device, batch_size)
      prediction = predict(execution.values)[:size].to(protocol.data.speed.device).realize()
      target = batch.target.to(protocol.data.speed.device).realize()
      yield (
        protocol.standardizer.restore(prediction) - protocol.standardizer.restore(target),
        batch.observed.to(protocol.data.speed.device).realize(),
      )

  return score(errors())


def _persistence(
  protocol: Protocol,
  span: WindowSpan,
  tensors: tuple[Tensor, Tensor, Tensor],
  samples: int,
  batch_size: int,
  shuffle: int,
) -> Scores:
  def errors():
    for batch in _sampled_batches(protocol, span, tensors, samples, batch_size, shuffle):
      target = batch.target.to(protocol.data.speed.device).realize()
      prediction = batch.anchor.expand(*target.shape).to(protocol.data.speed.device).realize()
      yield (
        protocol.standardizer.restore(prediction) - protocol.standardizer.restore(target),
        batch.observed.to(protocol.data.speed.device).realize(),
      )

  return score(errors())


def _sampled_batches(
  protocol: Protocol,
  span: WindowSpan,
  tensors: tuple[Tensor, Tensor, Tensor],
  samples: int,
  batch_size: int,
  shuffle: int,
) -> Iterator[WindowBatch]:
  yield from islice(
    batches(protocol, span, batch_size, shuffle=shuffle, tensors=tensors),
    samples // batch_size,
  )


def _probe_predictor(
  model: nn.Linear,
  represent: Representation,
  mean: Tensor,
  scale: Tensor,
) -> TinyJit:
  @TinyJit
  def predict(values: Tensor) -> Tensor:
    features = (represent(values) - mean) / scale
    return model(features).realize()

  return predict


def _raw_history(values: Tensor) -> Tensor:
  return values.permute(0, 2, 1, 3).reshape(values.shape[0], values.shape[2], -1).contiguous()


def _embedding(encoder: Encoder, diffusion: DirectedDiffusion, values: Tensor) -> Tensor:
  state = encoder(values, diffusion)
  return state.permute(0, 2, 1, 3).reshape(values.shape[0], values.shape[2], -1).contiguous()


def _blocks(batch: WindowBatch) -> tuple[Tensor, Tensor]:
  future = batch.target.permute(0, 2, 1).unsqueeze(-1).cat(
    batch.observed.permute(0, 2, 1).cast(batch.values.dtype).unsqueeze(-1),
    dim=-1,
  )
  return batch.values[..., :2].contiguous(), future.contiguous()


def _variation(values: Tensor) -> float:
  return values.std(axis=0).mean().item()


def _snapshot(model: object) -> dict[str, Tensor]:
  return {name: value.detach().clone().realize() for name, value in nn.state.get_state_dict(model).items()}


def _update_target(online: Encoder, target: Encoder, decay: float) -> None:
  source = nn.state.get_state_dict(online)
  Tensor.realize(*source.values())
  for name, value in nn.state.get_state_dict(target).items():
    value.assign(decay * value + (1 - decay) * source[name].detach()).realize()


def _arm(name: str, operators: dict[str, DirectedDiffusion]) -> tuple[str, bool, DirectedDiffusion]:
  topology = {"factorized": "true", "permuted": "permuted", "temporal": "self", "spatial": "true"}.get(name)
  if topology is None:
    raise ValueError(f"unknown arm {name!r}")
  return topology, name != "spatial", operators[topology]


def _spatial(encoder: Encoder) -> SpatialEncoder:
  return encoder.spatial if isinstance(encoder, FactorizedEncoder) else encoder


def _sparse_calls(model: Model, context: Tensor, target: Tensor, diffusion: DirectedDiffusion) -> int:
  return sum(
    uop.src[0].arg.name == "csr_sum"
    for uop in model.loss(context, target, diffusion).uop.toposort()
    if uop.op is Ops.CALL
  )


def _validate(protocol: Protocol, **settings: int | float | bool) -> None:
  if protocol.feature_names[:2] != ("speed", "observed"):
    raise ValueError("factorized JEPA requires speed and observed features")
  for name in ("steps", "probe_steps", "probe_samples", "evaluation_samples", "batch_size", "hidden_features"):
    value = settings[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      raise ValueError(f"{name} must be a positive integer")
  if settings["hidden_features"] < 2:
    raise ValueError("hidden_features must be at least two")
  if not isinstance(settings["seed"], int) or isinstance(settings["seed"], bool) or settings["seed"] < 0:
    raise ValueError("seed must be a non-negative integer")
  if settings["probe_samples"] > protocol.train.windows:
    raise ValueError("probe_samples must not exceed training windows")
  if settings["evaluation_samples"] > min(protocol.validation.windows, protocol.test.windows):
    raise ValueError("evaluation_samples must not exceed validation or test windows")
  for name in ("probe_samples", "evaluation_samples"):
    if settings[name] % settings["batch_size"]:
      raise ValueError(f"{name} must be divisible by batch_size")
    if settings["batch_size"] > settings[name]:
      raise ValueError(f"batch_size must not exceed {name}")
  if settings["learning_rate"] <= 0 or settings["probe_learning_rate"] <= 0:
    raise ValueError("learning rates must be positive")
  if not 0 <= settings["ema_decay"] < 1:
    raise ValueError("EMA decay must be in [0, 1)")
  if not isinstance(settings["evaluate_test"], bool):
    raise ValueError("evaluate_test must be boolean")


def main() -> None:
  observation = compare(
    Device.DEFAULT,
    seed=getenv("SEED", 0),
    history=getenv("HISTORY", 12),
    horizon=getenv("HORIZON", 12),
    steps=getenv("STEPS", 100),
    probe_steps=getenv("PROBE_STEPS", 100),
    probe_samples=getenv("SAMPLES", 512),
    evaluation_samples=getenv("EVAL_SAMPLES", 512),
    batch_size=getenv("BS", 64),
    hidden_features=getenv("HIDDEN", 8),
    learning_rate=getenv("LR", 0.001),
    ema_decay=getenv("EMA", 0.998),
    probe_learning_rate=getenv("PROBE_LR", 0.01),
    evaluate_test=bool(getenv("TEST", 0)),
  )
  print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
  main()
