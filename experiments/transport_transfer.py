"""Test whether one learned transport model transfers across graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from tinygrad import Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.transport_forecast import (
  DATA_SEED,
  NODES,
  TRAIN_TRAJECTORIES,
  VALIDATION_TRAJECTORIES,
  Checkpoint,
  Evaluation,
  Model,
  Operator,
  Trajectories,
  _fit,
  _metrics,
  _model,
  _operator,
  _parameter_count,
  _persistence,
  _predict,
  _topology,
  _trajectories,
)


TRAIN_HORIZON = 4
TRANSFER_SEED = DATA_SEED + 100
TRANSFER_TRAJECTORIES = 8
TARGET_NODES = (24, 32, 48)


@dataclass(frozen=True)
class Result:
  topology: str
  edges: int
  metrics: Evaluation


@dataclass(frozen=True)
class InitialResult:
  initial: str
  seed: int
  persistence: Evaluation
  models: tuple[Result, ...]


@dataclass(frozen=True)
class Observation:
  device: str
  data_seed: int
  transfer_seed: int
  seed: int
  model: str
  source_nodes: int
  source_edges: int
  train_trajectories: int
  validation_trajectories: int
  transfer_trajectories: int
  train_steps: int
  history: int
  horizon: int
  batch_size: int
  hidden_features: int
  epochs: int
  learning_rate: float
  parameters: int
  best_epoch: int
  training_seconds: float
  state: str
  checkpoints: tuple[Checkpoint, ...]
  target_nodes: int
  initial: InitialResult


def study(
  device: str,
  *,
  model_name: str = "diffusion_gru",
  target_nodes: int = 32,
  initial: str = "dense",
  seed: int = 0,
  epochs: int = 30,
  history: int = 4,
  horizon: int = 16,
  batch_size: int = 64,
  hidden_features: int = 8,
  learning_rate: float = 0.01,
) -> Observation:
  _validate(
    model_name,
    target_nodes=target_nodes,
    initial=initial,
    seed=seed,
    epochs=epochs,
    history=history,
    horizon=horizon,
    batch_size=batch_size,
    hidden_features=hidden_features,
    learning_rate=learning_rate,
  )
  source = _topology(NODES)
  steps = history + TRAIN_HORIZON
  train = _trajectories(source, TRAIN_TRAJECTORIES, steps, DATA_SEED, device)
  validation = _trajectories(
    source,
    VALIDATION_TRAJECTORIES,
    steps,
    DATA_SEED + 1,
    device,
  )
  Tensor.manual_seed(seed)
  model, operator, _ = _model(
    model_name,
    "none" if model_name == "lstm" else "true",
    source,
    hidden_features,
    device,
  )
  best_epoch, training_seconds, checkpoints = _fit(
    model,
    operator,
    train,
    validation,
    epochs=epochs,
    history=history,
    batch_size=batch_size,
    learning_rate=learning_rate,
  )
  state = _state(model)
  result = _scope(
    model,
    model_name,
    target_nodes,
    initial,
    device,
    history=history,
    horizon=horizon,
  )
  if _state(model) != state:
    raise RuntimeError("transfer evaluation mutated frozen model state")
  return Observation(
    device=device,
    data_seed=DATA_SEED,
    transfer_seed=TRANSFER_SEED,
    seed=seed,
    model=model_name,
    source_nodes=NODES,
    source_edges=len(source.source),
    train_trajectories=TRAIN_TRAJECTORIES,
    validation_trajectories=VALIDATION_TRAJECTORIES,
    transfer_trajectories=TRANSFER_TRAJECTORIES,
    train_steps=steps,
    history=history,
    horizon=horizon,
    batch_size=batch_size,
    hidden_features=hidden_features,
    epochs=epochs,
    learning_rate=learning_rate,
    parameters=_parameter_count(model),
    best_epoch=best_epoch,
    training_seconds=training_seconds,
    state=state,
    checkpoints=checkpoints,
    target_nodes=target_nodes,
    initial=result,
  )


def _scope(
  model: Model,
  model_name: str,
  nodes: int,
  initial: str,
  device: str,
  *,
  history: int,
  horizon: int,
) -> InitialResult:
  topology = _topology(nodes)
  structures = ("none",) if model_name == "lstm" else ("true", "permuted", "self")
  operators = []
  for structure in structures:
    operator, edges = _operator(model_name, structure, topology, device)
    operators.append((structure, operator, edges, _predictor(model, operator)))

  seed = TRANSFER_SEED + nodes + int(initial == "pulse")
  data = _trajectories(
    topology,
    TRANSFER_TRAJECTORIES,
    history + horizon,
    seed,
    device,
    initial=initial,
  )
  return InitialResult(
    initial,
    seed,
    _persistence(data, history, horizon),
    tuple(
      Result(
        structure,
        edges,
        _evaluate(
          model,
          operator,
          predict,
          data,
          history,
          horizon,
        ),
      )
      for structure, operator, edges, predict in operators
    )
  )


def _predictor(model: Model, operator: Operator) -> TinyJit:
  @TinyJit
  def predict(values: Tensor) -> Tensor:
    return _predict(model, values, operator).realize()

  return predict


def _evaluate(
  model: Model,
  operator: Operator,
  predict: TinyJit,
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
    prediction = predict(window.contiguous())
    predictions.append(prediction)
    window = window[:, 1:].cat(prediction.unsqueeze(1), dim=1).realize()
  rollout = Tensor.stack(*predictions, dim=1)
  expected = data.values[:, history : history + horizon]
  return Evaluation(
    one_step,
    _metrics(rollout, expected),
    _metrics(rollout[:, -1], expected[:, -1]),
  )


def _state(model: Model) -> str:
  values = {name: value.tolist() for name, value in nn.state.get_state_dict(model).items()}
  return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate(model: str, **settings: int | float | str) -> None:
  if model not in ("lstm", "diffusion_gru"):
    raise ValueError("model must be 'lstm' or 'diffusion_gru'")
  if settings["initial"] not in ("dense", "pulse"):
    raise ValueError("initial must be 'dense' or 'pulse'")
  for name in ("seed", "target_nodes", "epochs", "history", "horizon", "batch_size", "hidden_features"):
    value = settings[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= (0 if name != "seed" else -1):
      raise ValueError(f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer")
  if settings["target_nodes"] not in TARGET_NODES:
    raise ValueError(f"target_nodes must be one of {TARGET_NODES}")
  if settings["learning_rate"] <= 0:
    raise ValueError("learning_rate must be positive")


def main() -> None:
  observation = study(
    Device.DEFAULT,
    model_name=getenv("MODEL", "diffusion_gru"),
    target_nodes=getenv("NODES", 32),
    initial=getenv("INITIAL", "dense"),
    seed=getenv("SEED", 0),
    epochs=getenv("EPOCHS", 30),
    history=getenv("HISTORY", 4),
    horizon=getenv("HORIZON", 16),
    batch_size=getenv("BS", 64),
    hidden_features=getenv("HIDDEN", 8),
    learning_rate=getenv("LR", 0.01),
  )
  print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
  main()
