"""Train event-native memory on the frozen MBTA task and topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from tinygrad import Context, Tensor, nn

from experiments.mbta_event_memory import ARMS, SEEDS, EventEncoder
from experiments.mbta_topology import Node
from experiments.tools.mbta_headway_task import _digest, _read, _write
from experiments.tools.mbta_topology import (
  HUBER_DELTA,
  Examples,
  ResidualModel,
  _baselines,
  _mask_metrics,
  _metrics,
  _predict,
  _route_metrics,
  _state,
  _tensors,
  _topology,
  build as build_topology,
)


HIDDEN = 4
TIME_FEATURES = 2
ENCODER_BATCH_SIZE = 256
ENCODER_SEQUENCE_LIMIT = 1_024
TRUNCATION = 8
HEAD_HIDDEN = 16
HEAD_STEPS = 250
HEAD_BATCH_SIZE = 4_096
HEAD_CHECKPOINT_EVERY = 50
LEARNING_RATE = 0.003
SEQUENCE_SEED = 20_260_823
REFERENCES = {
  "submodules/pytorch-geometric": "5c6461b2305ad068a6d61165b3c55852a11aaa41",
  "submodules/pytorch-geometric-temporal": "fe555bc30ee197755c4b58a89407033a5f383415",
  "submodules/tinygrad": "33755a34657d25920914badbe32a9d70489669c7",
}
EXPECTED_ARTIFACTS = {
  "source_manifest_sha256": "780076ebf32ceb735b334b5bbdba0297d809088f0689fac37067db46fd36d701",
  "population_audit_sha256": "0345659ab2aa2a4c3341954dcfd789110e9567097f69532d25f895e7fa16e1bb",
  "task_protocol_sha256": "83e048c9f6e1f7f5de1ecb70e9a0a5cee64ac699b3ce3da94f4a9cd8f60e69ed",
  "topology_protocol_sha256": "e3c44f17757820eea46c7c6e066fe914991c920388e1a56a4e17f7c013303db8",
  "topology_validation_sha256": "9e3fb262e8be83314f0606b2553011d6a0a0a6d5ec96b24366334719048c3069",
  "topology_test_sha256": "d2cf0d52bbb0aabba48fc25b561a49bfc7102feb2865764f469e94d0133c9e9a",
  "clock_audit_sha256": "79755025ce1cc343fa59fedd2c34391dafa38bacf357564e00f93a5517234276",
}


class ExperimentError(ValueError):
  """The event-memory experiment violates a frozen fact or lifecycle."""


@dataclass(frozen=True)
class SplitData:
  examples: Examples
  shared: Any
  service_date: Any
  node: Any
  timestamp: Any
  cutoff: Any
  sequences: tuple[Any, ...]
  elapsed: Any
  anchor: Any
  target: Any
  target_mean: float
  target_scale: float


@dataclass(frozen=True)
class Encoded:
  before: Any
  after: Any
  prediction: Any


def build(
  source_dir: Path,
  population_audit: Path,
  task_protocol_path: Path,
  topology_protocol_path: Path,
  topology_validation_path: Path,
  topology_test_path: Path,
  clock_audit_path: Path,
) -> tuple[Any, dict]:
  paths = {
    "source_manifest_sha256": source_dir / "manifest.json",
    "population_audit_sha256": population_audit,
    "task_protocol_sha256": task_protocol_path,
    "topology_protocol_sha256": topology_protocol_path,
    "topology_validation_sha256": topology_validation_path,
    "topology_test_sha256": topology_test_path,
    "clock_audit_sha256": clock_audit_path,
  }
  digests = _verify(paths)
  connection, rebuilt = build_topology(source_dir, population_audit, task_protocol_path)
  frozen = _read(topology_protocol_path)
  if rebuilt != frozen:
    raise ExperimentError("topology: rebuilt protocol does not match frozen facts")
  if connection.execute(
    """
    SELECT count(*) FROM (
      SELECT service_date, parent_station, trunk_route_id, direction_id,
        departure_timestamp, count(*) AS targets
      FROM feature_base GROUP BY ALL HAVING targets != 1
    )
    """
  ).fetchone()[0]:
    raise ExperimentError("events: task contains a repeated lane timestamp")

  nodes, edges = _topology(connection)
  counts = connection.execute(
    """
    SELECT split, count(*) AS targets,
      count(DISTINCT (service_date, parent_station, trunk_route_id, direction_id)) AS lane_days,
      count(DISTINCT (service_date, departure_timestamp)) AS timestamp_groups
    FROM feature_base GROUP BY split ORDER BY split
    """
  ).fetchall()
  maximum = connection.execute(
    """
    SELECT max(events) FROM (
      SELECT count(*) AS events FROM feature_base
      GROUP BY split, service_date, parent_station, trunk_route_id, direction_id
    )
    """
  ).fetchone()[0]
  task = _read(task_protocol_path)
  selected_sequences = _selected_sequence_count(connection)
  return connection, {
    "schema": 1,
    **digests,
    "availability": task["availability"],
    "target": task["target"],
    "cutoff": task["cutoff"],
    "splits": task["splits"],
    "target_masks": task["target_masks"],
    "targets": sum(row[1] for row in counts),
    "nodes": len(nodes),
    "true_edges": len(edges),
    "route_blend": frozen["route_blend"],
    "arms": list(ARMS),
    "seeds": list(SEEDS),
    "prediction_order": "read_before_event_update",
    "memory_read_cutoff": "frozen_source_timestamp_plus_one_second",
    "equal_time_policy": "all_reads_then_deterministic_updates:no_cross_visibility",
    "state_reset": "every_service_day_lane_and_independent_replay",
    "state_bound": "O(nodes*hidden)+O(edges):no_event_pair_or_clock_carrier",
    "memory_update": "lane_local_gru_after_observed_headway",
    "topology_read": "affinity_weighted_latest_upstream_memory_at_strictly_earlier_timestamp",
    "split_counts": [
      {"split": split, "targets": targets, "lane_days": lane_days, "timestamp_groups": groups}
      for split, targets, lane_days, groups in counts
    ],
    "maximum_lane_day_events": maximum,
    "encoder": {
      "hidden_features": HIDDEN,
      "time_features": TIME_FEATURES,
      "time_encoding": "cosine_of_learned_normalized_log1p_elapsed_projection",
      "truncation_events": TRUNCATION,
      "batch_lane_days": ENCODER_BATCH_SIZE,
      "selected_train_lane_days": selected_sequences,
      "selection_seed": SEQUENCE_SEED,
      "selection": "validation_micro_mae_seconds_after_one_bounded_pass",
      "objective": f"huber_delta_{HUBER_DELTA}_standardized_log1p_seconds",
    },
    "head": {
      "architecture": "zero_initialized_residual_mlp",
      "hidden_features": HEAD_HIDDEN,
      "steps": HEAD_STEPS,
      "batch_size": HEAD_BATCH_SIZE,
      "learning_rate": LEARNING_RATE,
      "checkpoint_every": HEAD_CHECKPOINT_EVERY,
      "selection": "validation_micro_mae_seconds",
      "shared_features": "frozen_stage_4_anchor_inputs_plus_local_and_asof_memory",
    },
    "test_policy": "confirmatory_reuse_after_validation_freeze",
    "references": REFERENCES,
  }


def validation(connection: Any, protocol: dict, topology_validation: dict) -> dict:
  nodes = _nodes(connection)
  data = {split: _split(connection, split, protocol, nodes) for split in ("train", "validation")}
  baselines = _baselines(data["validation"].examples)
  results, encoders = [], []
  for seed in SEEDS:
    print(json.dumps({"phase": "encoder", "seed": seed}), flush=True)
    encoder, record = _fit_encoder(seed, data["train"], data["validation"])
    encoded = {split: _encode(encoder, values) for split, values in data.items()}
    record["state"] = _state(encoder)
    encoders.append(record)
    for arm in ARMS:
      print(json.dumps({"phase": "head", "seed": seed, "arm": arm}), flush=True)
      train_features, train_work = _arm_features(connection, arm, nodes, data["train"], encoded["train"])
      held_features, held_work = _arm_features(
        connection, arm, nodes, data["validation"], encoded["validation"]
      )
      result = _fit_head(
        arm,
        seed,
        data["train"].examples,
        data["validation"].examples,
        train_features,
        held_features,
      )
      result["replay_work"] = {"train": train_work, "validation": held_work}
      results.append(result)
      del train_features, held_features

  gate = _validation_gate(results, baselines, topology_validation)
  return {
    "schema": 1,
    "split": "validation",
    "targets": len(data["validation"].target),
    "protocol_sha256": _digest(protocol),
    "baselines": baselines,
    "encoders": encoders,
    "results": results,
    "validation_gate": gate,
    "decision": "freeze:confirm_event_memory_on_reused_test" if gate["passed"] else "stop:event_memory",
  }


def test(connection: Any, protocol: dict, frozen: dict, topology_test: dict) -> dict:
  if frozen.get("decision") != "freeze:confirm_event_memory_on_reused_test":
    raise ExperimentError("test: validation did not authorize confirmatory reuse")
  nodes = _nodes(connection)
  held = _split(connection, "test", protocol, nodes)
  baselines = _baselines(held.examples)
  results = []
  for record in frozen["encoders"]:
    print(json.dumps({"phase": "test_encoder", "seed": record["seed"]}), flush=True)
    encoder = EventEncoder(HIDDEN, TIME_FEATURES)
    nn.state.load_state_dict(encoder, _tensors(record["state"]), verbose=False)
    encoded = _encode(encoder, held)
    for arm in ARMS:
      print(json.dumps({"phase": "test_head", "seed": record["seed"], "arm": arm}), flush=True)
      features, work = _arm_features(connection, arm, nodes, held, encoded)
      trained = next(row for row in frozen["results"] if (row["arm"], row["seed"]) == (arm, record["seed"]))
      result = _evaluate_head(trained, held.examples, features)
      result["replay_work"] = {"test": work}
      results.append(result)
  gate = _test_gate(results, baselines, frozen, topology_test)
  return {
    "schema": 1,
    "split": "test",
    "targets": len(held.target),
    "protocol_sha256": _digest(protocol),
    "validation_sha256": _digest(frozen),
    "test_policy": "confirmatory_reuse:not_pristine_holdout",
    "baselines": baselines,
    "results": results,
    "claim_gate": gate,
    "decision": "retain:event_memory" if gate["passed"] else "stop:event_memory",
  }


def _split(connection: Any, split: str, protocol: dict, nodes: tuple[Node, ...]) -> SplitData:
  import numpy as np

  from experiments.tools.mbta_topology import _examples

  examples = _examples(connection, split, protocol)
  metadata = connection.execute(
    """
    SELECT service_date, parent_station, trunk_route_id, direction_id,
      departure_timestamp, cutoff_timestamp
    FROM feature_base WHERE split = ? ORDER BY target_id
    """,
    [split],
  ).fetchall()
  if len(metadata) != len(examples.target):
    raise ExperimentError(f"{split}: metadata and target counts differ")
  node_ids = {node: index for index, node in enumerate(nodes)}
  service_date = np.asarray([row[0] for row in metadata], dtype="int32")
  node = np.asarray([node_ids[Node(row[1], row[2], int(row[3]))] for row in metadata], dtype="int16")
  timestamp = np.asarray([row[4] for row in metadata], dtype="int64")
  cutoff = np.asarray([row[5] for row in metadata], dtype="int64")
  groups: dict[tuple[int, int], list[int]] = {}
  for index, key in enumerate(zip(service_date, node)):
    groups.setdefault(key, []).append(index)
  sequences = tuple(
    np.asarray(sorted(indexes, key=timestamp.__getitem__), dtype="int32")
    for _, indexes in sorted(groups.items())
  )
  logarithm = np.log1p(examples.target.astype("float64"))
  target_mean, target_scale = protocol["encoder_scaler"]["mean"], protocol["encoder_scaler"]["scale"]
  elapsed = ((logarithm - target_mean) / target_scale).astype("float32")[:, None]
  target = elapsed[:, 0]
  anchor = ((np.log1p(examples.anchor) - target_mean) / target_scale).astype("float32")
  base = examples.arm_features["self"]
  shared = np.concatenate((base[:, :9], base[:, 12:]), axis=1).astype("float32")
  return SplitData(
    examples,
    shared,
    service_date,
    node,
    timestamp,
    cutoff,
    sequences,
    elapsed,
    anchor,
    target,
    target_mean,
    target_scale,
  )


def _fit_encoder(seed: int, train: SplitData, held: SplitData) -> tuple[EventEncoder, dict]:
  import numpy as np

  Tensor.manual_seed(seed)
  model = EventEncoder(HIDDEN, TIME_FEATURES)
  initial_state = _state(model)
  initial_mae = float(
    np.abs(np.expm1(held.anchor * held.target_scale + held.target_mean) - held.examples.target).mean()
  )
  checkpoints = [{"pass": 0, "mae_seconds": round(initial_mae, 6)}]
  parameters = nn.state.get_parameters(model)
  optimizer = nn.optim.Adam(parameters, lr=LEARNING_RATE, fused=False)
  chosen = list(train.sequences)
  Random(SEQUENCE_SEED).shuffle(chosen)
  chosen = chosen[:ENCODER_SEQUENCE_LIMIT]
  Random(seed).shuffle(chosen)
  trained = sum(len(sequence) for sequence in chosen)
  for start in range(0, len(chosen), ENCODER_BATCH_SIZE):
    _train_sequence_batch(model, optimizer, chosen[start:start + ENCODER_BATCH_SIZE], train)
  prediction = _encode(model, held).prediction
  mae = float(np.abs(prediction - held.examples.target).mean())
  checkpoints.append({"pass": 1, "mae_seconds": round(mae, 6)})
  selected = 1 if mae < initial_mae else 0
  if selected == 0:
    nn.state.load_state_dict(model, _tensors(initial_state), verbose=False)
  return model, {
    "seed": seed,
    "parameters": sum(parameter.numel() for parameter in parameters),
    "selected_pass": selected,
    "selected_train_lane_days": len(chosen),
    "trained_events": trained,
    "checkpoints": checkpoints,
  }


def _train_sequence_batch(model: EventEncoder, optimizer: Any, sequences: list[Any], data: SplitData) -> None:
  import numpy as np

  ids = _padded(sequences)
  state = Tensor.zeros(len(sequences), HIDDEN)
  for start in range(0, ids.shape[1], TRUNCATION):
    stop = min(start + TRUNCATION, ids.shape[1])
    loss = Tensor.zeros(())
    count = 0
    with Context(TRAINING=1):
      optimizer.zero_grad()
      for step in range(start, stop):
        indexes = ids[:, step]
        mask = indexes >= 0
        safe = np.maximum(indexes, 0)
        observed = Tensor(data.elapsed[safe])
        prediction = model.predict(state, Tensor(data.anchor[safe]))
        error = prediction - Tensor(data.target[safe])
        absolute = error.abs()
        item = (absolute < HUBER_DELTA).where(
          error.square() / (2 * HUBER_DELTA),
          absolute - HUBER_DELTA / 2,
        )
        loss = loss + Tensor(mask).where(item, 0).sum()
        count += int(mask.sum())
        update = model.update(observed, state)
        state = Tensor(mask[:, None]).where(update, state)
      (loss / count).backward()
      loss.realize(*optimizer.schedule_step())
    state = Tensor(state.numpy())


def _encode(model: EventEncoder, data: SplitData) -> Encoded:
  import numpy as np

  before = np.zeros((len(data.target), HIDDEN), dtype="float32")
  after = np.zeros_like(before)
  prediction = np.zeros(len(data.target), dtype="float32")
  weights = {name: value.numpy() for name, value in nn.state.get_state_dict(model).items()}
  for start in range(0, len(data.sequences), ENCODER_BATCH_SIZE):
    sequences = data.sequences[start:start + ENCODER_BATCH_SIZE]
    ids = _padded(sequences)
    state = np.zeros((len(sequences), HIDDEN), dtype="float32")
    for step in range(ids.shape[1]):
      indexes = ids[:, step]
      mask = indexes >= 0
      selected = indexes[mask]
      before[selected] = state[mask]
      prediction[selected] = (
        data.anchor[selected]
        + state[mask] @ weights["output.weight"].T[:, 0]
        + weights["output.bias"][0]
      )
      state[mask] = _encoder_step(data.elapsed[selected], state[mask], weights)
      after[selected] = state[mask]
  restored = np.expm1(prediction * data.target_scale + data.target_mean)
  return Encoded(before, after, restored)


def _encoder_step(elapsed: Any, state: Any, weights: dict[str, Any]) -> Any:
  import numpy as np

  time = np.cos(elapsed @ weights["time.weight"].T + weights["time.bias"])
  encoded = np.concatenate((elapsed, time), axis=1)
  joined = np.concatenate((encoded, state), axis=1)
  gates = joined @ weights["gates.weight"].T + weights["gates.bias"]
  update = 1 / (1 + np.exp(-gates[:, :HIDDEN]))
  reset = 1 / (1 + np.exp(-gates[:, HIDDEN:]))
  candidate_input = np.concatenate((encoded, state * reset), axis=1)
  candidate = np.tanh(candidate_input @ weights["candidate.weight"].T + weights["candidate.bias"])
  return (update * state + (1 - update) * candidate).astype("float32")


def _arm_features(
  connection: Any,
  arm: str,
  nodes: tuple[Node, ...],
  data: SplitData,
  encoded: Encoded,
) -> tuple[Any, dict]:
  import numpy as np

  messages = np.zeros_like(encoded.before)
  local = np.zeros_like(encoded.before)
  ages = np.zeros(len(data.target), dtype="float32")
  observed = np.zeros(len(data.target), dtype="float32")
  incidences = 0
  incoming = None if arm == "node_local" else _incoming(connection, arm, nodes)
  state = np.zeros((len(nodes), HIDDEN), dtype="float32")
  last = np.full(len(nodes), -1, dtype="int64")
  event_order = np.lexsort((data.node, data.timestamp, data.service_date))
  query_order = np.lexsort((data.node, data.cutoff, data.service_date))
  event_days = dict(_groups(event_order, data.service_date))
  for day, queries in _groups(query_order, data.service_date):
    state.fill(0)
    last.fill(-1)
    events = event_days[day]
    cursor = 0
    for cutoff, query_indexes in _groups(queries, data.cutoff):
      while cursor < len(events) and data.timestamp[events[cursor]] < cutoff:
        event = events[cursor]
        event_node = int(data.node[event])
        state[event_node] = encoded.after[event]
        last[event_node] = data.timestamp[event]
        cursor += 1
      for index in query_indexes:
        target_node = int(data.node[index])
        if last[target_node] >= 0:
          local[index] = state[target_node]
        if incoming is None:
          continue
        edges = [(source, weight) for source, weight in incoming[target_node] if last[source] >= 0]
        if not edges:
          continue
        weight = sum(value for _, value in edges)
        messages[index] = sum((state[source] * value for source, value in edges), np.zeros(HIDDEN)) / weight
        ages[index] = sum((cutoff - last[source]) * value for source, value in edges) / weight
        observed[index] = len(edges)
        incidences += len(edges)
  if not np.allclose(local, encoded.before, atol=1e-6):
    raise ExperimentError("replay: cutoff-local state differs from the lane fold")
  features = np.concatenate(
    (
      data.shared,
      local,
      messages,
      np.log1p(ages)[:, None],
      (observed > 0).astype("float32")[:, None],
    ),
    axis=1,
  ).astype("float32")
  return features, {
    "targets": len(data.target),
    "observed_messages": int((observed > 0).sum()),
    "target_edge_incidences": incidences,
    "persistent_state_values": len(nodes) * HIDDEN,
  }


def _fit_head(
  arm: str,
  seed: int,
  train: Examples,
  held: Examples,
  train_features: Any,
  held_features: Any,
) -> dict:
  import numpy as np

  mean = train_features.mean(axis=0)
  scale = train_features.std(axis=0)
  scale[scale < 1e-6] = 1
  target = np.log1p(train.target)
  target_mean, target_scale = float(target.mean()), float(target.std())
  training = ((train_features - mean) / scale).astype("float32")
  validation = ((held_features - mean) / scale).astype("float32")
  normalized_target = ((target - target_mean) / target_scale).astype("float32")
  normalized_anchor = ((np.log1p(train.anchor) - target_mean) / target_scale).astype("float32")
  held_anchor = ((np.log1p(held.anchor) - target_mean) / target_scale).astype("float32")
  Tensor.manual_seed(seed)
  model = ResidualModel(training.shape[1], HEAD_HIDDEN)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE, fused=False)
  random = np.random.default_rng(seed)
  initial = _predict(model, validation, held_anchor, target_mean, target_scale)
  best_mae, best_step, best_state = float(np.abs(initial - held.target).mean()), 0, _state(model)
  checkpoints = [{"step": 0, "mae_seconds": round(best_mae, 6)}]
  for step in range(1, HEAD_STEPS + 1):
    index = random.integers(0, len(training), size=HEAD_BATCH_SIZE)
    with Context(TRAINING=1):
      optimizer.zero_grad()
      error = model(Tensor(training[index]), Tensor(normalized_anchor[index])) - Tensor(normalized_target[index])
      absolute = error.abs()
      loss = (absolute < HUBER_DELTA).where(
        error.square() / (2 * HUBER_DELTA),
        absolute - HUBER_DELTA / 2,
      ).mean().backward()
      loss.realize(*optimizer.schedule_step())
    if step % HEAD_CHECKPOINT_EVERY:
      continue
    prediction = _predict(model, validation, held_anchor, target_mean, target_scale)
    mae = float(np.abs(prediction - held.target).mean())
    checkpoints.append({"step": step, "mae_seconds": round(mae, 6)})
    if mae < best_mae:
      best_mae, best_step, best_state = mae, step, _state(model)
  nn.state.load_state_dict(model, _tensors(best_state), verbose=False)
  prediction = _predict(model, validation, held_anchor, target_mean, target_scale)
  return {
    "arm": arm,
    "seed": seed,
    "best_step": best_step,
    "parameters": sum(parameter.numel() for parameter in nn.state.get_parameters(model)),
    "trained_examples": HEAD_STEPS * HEAD_BATCH_SIZE,
    "checkpoints": checkpoints,
    "metrics": _metrics(held, prediction),
    "route_metrics": _route_metrics(held, prediction),
    "mask_metrics": _mask_metrics(held, prediction),
    "scaler": {
      "feature_mean": mean.tolist(),
      "feature_scale": scale.tolist(),
      "target_mean": target_mean,
      "target_scale": target_scale,
    },
    "state": best_state,
  }


def _evaluate_head(frozen: dict, examples: Examples, features: Any) -> dict:
  import numpy as np

  scaler = frozen["scaler"]
  values = ((features - np.asarray(scaler["feature_mean"])) / np.asarray(scaler["feature_scale"])).astype("float32")
  anchor = ((np.log1p(examples.anchor) - scaler["target_mean"]) / scaler["target_scale"]).astype("float32")
  model = ResidualModel(values.shape[1], HEAD_HIDDEN)
  nn.state.load_state_dict(model, _tensors(frozen["state"]), verbose=False)
  prediction = _predict(model, values, anchor, scaler["target_mean"], scaler["target_scale"])
  return {
    "arm": frozen["arm"],
    "seed": frozen["seed"],
    "best_step": frozen["best_step"],
    "parameters": frozen["parameters"],
    "trained_examples": frozen["trained_examples"],
    "metrics": _metrics(examples, prediction),
    "route_metrics": _route_metrics(examples, prediction),
    "mask_metrics": _mask_metrics(examples, prediction),
  }


def _validation_gate(results: list[dict], baselines: dict, incumbent: dict) -> dict:
  incumbent_mae = _mean_result(incumbent["results"], "true", "mae_seconds")
  means = {
    metric: {arm: _mean_result(results, arm, metric) for arm in ARMS}
    for metric in ("mae_seconds", "macro_route_mae_seconds", "p90_absolute_error")
  }
  selected = min(ARMS, key=lambda arm: (means["mae_seconds"][arm], arm))
  seed_passed = all(
    _result(results, selected, seed)["metrics"]["mae_seconds"]
    < _result(incumbent["results"], "true", seed)["metrics"]["mae_seconds"]
    for seed in SEEDS
  )
  topology_controls = ("node_local", "self", "reverse", "permuted")
  topology_seed_passed = all(
    _result(results, "true", seed)["metrics"]["mae_seconds"]
    < min(_result(results, arm, seed)["metrics"]["mae_seconds"] for arm in topology_controls)
    for seed in SEEDS
  )
  topology_macro_passed = means["macro_route_mae_seconds"]["true"] < min(
    means["macro_route_mae_seconds"][arm] for arm in topology_controls
  )
  topology_tail_passed = means["p90_absolute_error"]["true"] < min(
    means["p90_absolute_error"][arm] for arm in topology_controls
  )
  passed = (
    means["mae_seconds"][selected] < incumbent_mae
    and means["mae_seconds"][selected]
    < min(baselines["temporal"]["mae_seconds"], baselines["plan"]["mae_seconds"])
    and seed_passed
  )
  return {
    "selected_arm": selected,
    "mean": {
      metric: {arm: round(value, 6) for arm, value in arms.items()}
      for metric, arms in means.items()
    },
    "incumbent_true_topology_mae_seconds": round(incumbent_mae, 6),
    "selected_beats_incumbent_in_every_seed": seed_passed,
    "true_beats_every_memory_control_in_every_seed": topology_seed_passed,
    "true_beats_memory_controls_on_macro_route_mae": topology_macro_passed,
    "true_beats_memory_controls_on_p90_absolute_error": topology_tail_passed,
    "passed": passed,
  }


def _test_gate(results: list[dict], baselines: dict, validation_evidence: dict, incumbent: dict) -> dict:
  selected = validation_evidence["validation_gate"]["selected_arm"]
  mean = _mean_result(results, selected, "mae_seconds")
  incumbent_mae = _mean_result(incumbent["results"], "true", "mae_seconds")
  seed_passed = all(
    _result(results, selected, seed)["metrics"]["mae_seconds"]
    < _result(incumbent["results"], "true", seed)["metrics"]["mae_seconds"]
    for seed in SEEDS
  )
  passed = (
    mean < incumbent_mae
    and mean < min(baselines["temporal"]["mae_seconds"], baselines["plan"]["mae_seconds"])
    and seed_passed
  )
  return {
    "selected_arm": selected,
    "selected_mean_mae_seconds": round(mean, 6),
    "incumbent_true_topology_mae_seconds": round(incumbent_mae, 6),
    "selected_beats_incumbent_in_every_seed": seed_passed,
    "passed": passed,
  }


def _nodes(connection: Any) -> tuple[Node, ...]:
  return tuple(
    Node(station, route, int(direction))
    for station, route, direction in connection.execute(
      "SELECT parent_station, trunk_route_id, direction_id FROM events GROUP BY ALL ORDER BY ALL"
    ).fetchall()
  )


def _incoming(connection: Any, arm: str, nodes: tuple[Node, ...]) -> tuple[tuple[tuple[int, float], ...], ...]:
  node_ids = {node: index for index, node in enumerate(nodes)}
  incoming: list[list[tuple[int, float]]] = [[] for _ in nodes]
  rows = connection.execute(
    f"""
    SELECT source_station, trunk_route_id, direction_id, target_station, affinity
    FROM {arm}_edges ORDER BY ALL
    """
  ).fetchall()
  for source, route, direction, target, affinity in rows:
    incoming[node_ids[Node(target, route, int(direction))]].append(
      (node_ids[Node(source, route, int(direction))], float(affinity))
    )
  return tuple(tuple(edges) for edges in incoming)


def _groups(order: Any, values: Any) -> Any:
  start = 0
  while start < len(order):
    stop = start + 1
    while stop < len(order) and values[order[stop]] == values[order[start]]:
      stop += 1
    yield values[order[start]], order[start:stop]
    start = stop


def _padded(sequences: list[Any] | tuple[Any, ...]) -> Any:
  import numpy as np

  width = max(map(len, sequences))
  result = np.full((len(sequences), width), -1, dtype="int32")
  for row, sequence in enumerate(sequences):
    result[row, :len(sequence)] = sequence
  return result


def _result(results: list[dict], arm: str, seed: int) -> dict:
  return next(row for row in results if (row["arm"], row["seed"]) == (arm, seed))


def _mean_result(results: list[dict], arm: str, metric: str) -> float:
  return sum(_result(results, arm, seed)["metrics"][metric] for seed in SEEDS) / len(SEEDS)


def _selected_sequence_count(connection: Any) -> int:
  counts = [row[0] for row in connection.execute(
    """
    SELECT count(*) FROM feature_base WHERE split = 'train'
    GROUP BY service_date, parent_station, trunk_route_id, direction_id ORDER BY ALL
    """
  ).fetchall()]
  chosen = list(counts)
  Random(SEQUENCE_SEED).shuffle(chosen)
  return min(len(chosen), ENCODER_SEQUENCE_LIMIT)


def _verify(paths: dict[str, Path]) -> dict[str, str]:
  try:
    observed = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
  except FileNotFoundError as error:
    raise ExperimentError(f"{error.filename}: missing frozen artifact") from error
  if observed != EXPECTED_ARTIFACTS:
    raise ExperimentError("artifacts: frozen digest drift")
  return observed


def _freeze(path: Path, value: object) -> None:
  encoded = json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
  if path.exists():
    if path.read_text() != encoded:
      raise ExperimentError(f"{path.name}: frozen artifact drift")
    return
  _write(path, value)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--population-audit", type=Path, required=True)
  parser.add_argument("--task-protocol", type=Path, required=True)
  parser.add_argument("--topology-protocol", type=Path, required=True)
  parser.add_argument("--topology-validation", type=Path, required=True)
  parser.add_argument("--topology-test", type=Path, required=True)
  parser.add_argument("--clock-audit", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--test", action="store_true")
  arguments = parser.parse_args()
  arguments.output_dir.mkdir(parents=True, exist_ok=True)
  protocol_path = arguments.output_dir / "protocol.json"
  validation_path = arguments.output_dir / "validation.json"
  test_path = arguments.output_dir / "test.json"
  if arguments.test and test_path.exists():
    raise ExperimentError("test.json: confirmatory split was already recomputed")
  connection, protocol = build(
    arguments.source_dir,
    arguments.population_audit,
    arguments.task_protocol,
    arguments.topology_protocol,
    arguments.topology_validation,
    arguments.topology_test,
    arguments.clock_audit,
  )
  import numpy as np

  train_target = connection.execute(
    "SELECT elapsed_seconds FROM feature_base WHERE split = 'train' ORDER BY target_id"
  ).fetchnumpy()["elapsed_seconds"].astype("float64")
  logarithm = np.log1p(train_target)
  protocol["encoder_scaler"] = {"mean": float(logarithm.mean()), "scale": float(logarithm.std())}
  if arguments.test:
    frozen_protocol = _read(protocol_path)
    frozen_validation = _read(validation_path)
    if protocol != frozen_protocol:
      raise ExperimentError("test: rebuilt memory protocol does not match frozen facts")
    evidence = test(connection, protocol, frozen_validation, _read(arguments.topology_test))
    _write(test_path, evidence)
  else:
    _freeze(protocol_path, protocol)
    evidence = validation(connection, protocol, _read(arguments.topology_validation))
    _freeze(validation_path, evidence)
  print(json.dumps({"decision": evidence["decision"], "split": evidence["split"]}))


if __name__ == "__main__":
  main()
