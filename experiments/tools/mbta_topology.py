"""Build matched causal features for the frozen MBTA topology experiment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Any

from tinygrad import Context, Tensor, nn

from experiments.mbta_topology import Edge, Node, ResidualModel, degree_sequence, permute_nodes, relabel
from experiments.tools.mbta_headway_task import _digest, _fit_temporal, _read, _rows, _write, open_task


ARMS = ("self", "true", "reverse", "permuted")
BLEND_CANDIDATES = tuple(index / 20 for index in range(21))
SEEDS = (0, 1, 2)
HIDDEN = 16
STEPS = 500
BATCH_SIZE = 4096
LEARNING_RATE = 0.003
CHECKPOINT_EVERY = 100
HUBER_DELTA = 0.25
EVALUATION_BATCH_SIZE = 16_384
REFERENCES = {
  "submodules/libcity": "5a6391d41944e937f2c15e9be85ab7f40ac8b23e",
  "submodules/pytorch-geometric-temporal": "fe555bc30ee197755c4b58a89407033a5f383415",
  "submodules/tinygrad": "33755a34657d25920914badbe32a9d70489669c7",
  "submodules/torch-spatiotemporal": "aa5f313e000d192bdec270748b8d01df5912e58e",
}


class ExperimentError(ValueError):
  """The feature population violates the frozen topology contract."""


def prepare(connection: Any, selected: dict) -> None:
  _fit_temporal(connection, selected["bin_hours"])
  _base(connection, selected["minimum_support"])
  nodes, edges = _topology(connection)
  _edge_tables(connection, nodes, edges)
  for arm in ARMS:
    _messages(connection, arm)
  connection.execute(
    """
    CREATE TEMP TABLE topology_features AS
    SELECT base.*,
      self_message.message_seconds AS self_message_seconds,
      self_message.age_seconds AS self_age_seconds,
      self_message.observed AS self_observed,
      true_message.message_seconds AS true_message_seconds,
      true_message.age_seconds AS true_age_seconds,
      true_message.observed AS true_observed,
      reverse_message.message_seconds AS reverse_message_seconds,
      reverse_message.age_seconds AS reverse_age_seconds,
      reverse_message.observed AS reverse_observed,
      permuted_message.message_seconds AS permuted_message_seconds,
      permuted_message.age_seconds AS permuted_age_seconds,
      permuted_message.observed AS permuted_observed
    FROM feature_base base
    LEFT JOIN self_messages self_message USING (target_id)
    LEFT JOIN true_messages true_message USING (target_id)
    LEFT JOIN reverse_messages reverse_message USING (target_id)
    LEFT JOIN permuted_messages permuted_message USING (target_id)
    """
  )
  actual = connection.execute("SELECT split, count(*) FROM topology_features GROUP BY split ORDER BY split").fetchall()
  expected = [("test", 176_568), ("train", 625_073), ("validation", 138_910)]
  if actual != expected:
    raise ExperimentError(f"features: target parity drift: {actual}")


def protocol(connection: Any, task_protocol: dict) -> dict:
  topology = {}
  for arm in ARMS:
    topology[arm] = _rows(
      connection,
      f"""
      SELECT count(*) AS edges,
        count(DISTINCT (source_station, trunk_route_id, direction_id)) AS source_nodes,
        count(DISTINCT (target_station, trunk_route_id, direction_id)) AS target_nodes,
        min(affinity) AS minimum_affinity, max(affinity) AS maximum_affinity,
        sum(affinity) AS total_affinity
      FROM {arm}_edges
      """,
    )[0]
  coverage = _rows(
    connection,
    """
    SELECT split,
      count(self_message_seconds) AS self,
      count(true_message_seconds) AS true,
      count(reverse_message_seconds) AS reverse,
      count(permuted_message_seconds) AS permuted,
      count(*) AS targets
    FROM topology_features GROUP BY split
    ORDER BY CASE split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END
    """,
  )
  blend = _rows(connection, "SELECT route_id, plan_weight FROM route_blend ORDER BY route_id")
  return {
    "schema": 1,
    "task_protocol_sha256": _digest(task_protocol),
    "availability": task_protocol["availability"],
    "target": task_protocol["target"],
    "cutoff": task_protocol["cutoff"],
    "splits": task_protocol["splits"],
    "targets": sum(row["targets"] for row in task_protocol["target_counts"]),
    "arms": list(ARMS),
    "topology_source": "train_only_exact_run_relations",
    "topology": topology,
    "message": "affinity_weighted_latest_exact_headway_per_incoming_neighbor_before_cutoff",
    "message_coverage": coverage,
    "anchor": "train_fitted_route_blend_of_temporal_and_public_plan",
    "blend_candidates": list(BLEND_CANDIDATES),
    "route_blend": blend,
    "continuous_features": [
      "log1p_temporal",
      "log1p_plan",
      "log1p_persistence",
      "plan_observed",
      "persistence_observed",
      "weekday_sine",
      "weekday_cosine",
      "hour_sine",
      "hour_cosine",
      "log1p_message",
      "log1p_message_age",
      "message_observed",
    ],
    "categorical_features": ["route_id"],
    "history": "latest_completed_exact_headway_within_target_service_date",
    "topology_seed": 20_260_822,
    "target_masks": task_protocol["target_masks"],
    "seeds": task_protocol["seeds"],
    "model": {
      "architecture": "zero_initialized_residual_mlp",
      "hidden_features": HIDDEN,
      "steps": STEPS,
      "batch_size": BATCH_SIZE,
      "learning_rate": LEARNING_RATE,
      "checkpoint_every": CHECKPOINT_EVERY,
      "objective": f"huber_delta_{HUBER_DELTA}_standardized_log1p_seconds",
      "selection": "validation_micro_mae_seconds",
      "coverage": "full:missing_plan_persistence_and_message_impute_to_declared_causal_controls_with_masks",
      "numpy": "2.3.2:one_off_feature_and_metric_arrays",
    },
    "references": REFERENCES,
  }


def build(source_dir: Path, population_audit: Path, task_protocol_path: Path) -> tuple[Any, dict]:
  connection, rebuilt, selected, _ = open_task(source_dir, population_audit)
  frozen = json.loads(task_protocol_path.read_bytes())
  if rebuilt != frozen:
    raise ExperimentError("task: frozen Stage 3 protocol does not match rebuilt source facts")
  connection.execute("SET threads = 1")
  prepare(connection, selected)
  return connection, protocol(connection, rebuilt)


def _base(connection: Any, minimum_support: int) -> None:
  connection.execute(
    """
    CREATE TEMP TABLE feature_targets AS
    SELECT row_number() OVER (
      ORDER BY target.service_date, target.vehicle_id, target.parent_station,
        target.direction_id, target.departure_timestamp
    ) AS target_id, target.*,
      coalesce(
        CASE WHEN cell.samples >= ? THEN cell.median_seconds END,
        CASE WHEN lane.samples >= ? THEN lane.median_seconds END,
        CASE WHEN route.samples >= ? THEN route.median_seconds END,
        global.median_seconds
      )::DOUBLE AS temporal_seconds
    FROM task target
    LEFT JOIN temporal_cell cell
      ON cell.parent_station = target.parent_station AND cell.trunk_route_id = target.trunk_route_id
      AND cell.direction_id = target.direction_id AND cell.weekday = target.weekday
      AND cell.hour_bin = target.local_hour
    LEFT JOIN temporal_lane lane
      ON lane.parent_station = target.parent_station AND lane.trunk_route_id = target.trunk_route_id
      AND lane.direction_id = target.direction_id
    LEFT JOIN temporal_route route ON route.route_id = target.route_id
    CROSS JOIN temporal_global global
    """,
    [minimum_support, minimum_support, minimum_support],
  )
  connection.execute("CREATE TEMP TABLE blend_candidates(plan_weight DOUBLE)")
  connection.executemany("INSERT INTO blend_candidates VALUES (?)", [(value,) for value in BLEND_CANDIDATES])
  connection.execute(
    """
    CREATE TEMP TABLE route_blend AS
    SELECT route_id, plan_weight
    FROM (
      SELECT targets.route_id, candidates.plan_weight,
        avg(abs(
          (1 - candidates.plan_weight) * targets.temporal_seconds
          + candidates.plan_weight * coalesce(targets.plan_seconds, targets.temporal_seconds)
          - targets.elapsed_seconds
        )) AS mae_seconds
      FROM feature_targets targets CROSS JOIN blend_candidates candidates
      WHERE targets.split = 'train'
      GROUP BY targets.route_id, candidates.plan_weight
    )
    QUALIFY row_number() OVER (PARTITION BY route_id ORDER BY mae_seconds, plan_weight) = 1
    """
  )
  connection.execute(
    """
    CREATE TEMP TABLE feature_base AS
    SELECT targets.*,
      coalesce(targets.plan_seconds, targets.temporal_seconds)::DOUBLE AS plan_value,
      (targets.plan_seconds IS NOT NULL)::INTEGER AS plan_observed,
      coalesce(targets.persistence_seconds, targets.temporal_seconds)::DOUBLE AS persistence_value,
      (targets.persistence_seconds IS NOT NULL)::INTEGER AS persistence_observed,
      ((1 - blend.plan_weight) * targets.temporal_seconds
        + blend.plan_weight * coalesce(targets.plan_seconds, targets.temporal_seconds))::DOUBLE AS anchor_seconds
    FROM feature_targets targets JOIN route_blend blend USING (route_id)
    """
  )


def _topology(connection: Any) -> tuple[tuple[Node, ...], tuple[Edge, ...]]:
  nodes = tuple(
    Node(parent_station, trunk_route_id, int(direction_id))
    for parent_station, trunk_route_id, direction_id in connection.execute(
      "SELECT parent_station, trunk_route_id, direction_id FROM events GROUP BY ALL ORDER BY ALL"
    ).fetchall()
  )
  edges = tuple(
    Edge(
      Node(source_station, trunk_route_id, int(direction_id)),
      Node(target_station, trunk_route_id, int(direction_id)),
      float(affinity),
    )
    for source_station, trunk_route_id, direction_id, target_station, affinity in connection.execute(
      """
      SELECT runs.parent_station, mapping.trunk_route_id, runs.direction_id,
        runs.target_parent_station, count(*) AS affinity
      FROM exact_run_relations runs JOIN route_mapping mapping USING (route_id)
      WHERE runs.service_date <= 20260810
        AND runs.direction_id = runs.target_direction_id
        AND runs.parent_station != runs.target_parent_station
      GROUP BY ALL ORDER BY ALL
      """
    ).fetchall()
  )
  known = set(nodes)
  if not edges or any(edge.source not in known or edge.target not in known for edge in edges):
    raise ExperimentError("topology: unresolved train edge")
  return nodes, edges


def _edge_tables(connection: Any, nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> None:
  reverse = tuple(Edge(edge.target, edge.source, edge.affinity) for edge in edges)
  self_edges = tuple(Edge(node, node, 1.0) for node in nodes)
  permuted = relabel(edges, permute_nodes(nodes))
  if degree_sequence(edges) != degree_sequence(permuted):
    raise ExperimentError("topology: permutation changed the degree sequence")
  for arm, values in zip(ARMS, (self_edges, edges, reverse, permuted)):
    connection.execute(
      f"""
      CREATE TEMP TABLE {arm}_edges(
        source_station VARCHAR, trunk_route_id VARCHAR, direction_id INTEGER,
        target_station VARCHAR, affinity DOUBLE
      )
      """
    )
    connection.executemany(
      f"INSERT INTO {arm}_edges VALUES (?, ?, ?, ?, ?)",
      [
        (
          edge.source.parent_station,
          edge.source.trunk_route_id,
          edge.source.direction_id,
          edge.target.parent_station,
          edge.affinity,
        )
        for edge in values
      ],
    )


def _messages(connection: Any, arm: str) -> None:
  connection.execute(
    f"""
    CREATE TEMP TABLE {arm}_pairs AS
    SELECT targets.target_id, targets.service_date, targets.cutoff_timestamp,
      edges.source_station, edges.trunk_route_id, edges.direction_id, edges.affinity
    FROM feature_base targets JOIN {arm}_edges edges
      ON edges.target_station = targets.parent_station
      AND edges.trunk_route_id = targets.trunk_route_id
      AND edges.direction_id = targets.direction_id
    """
  )
  connection.execute(
    f"""
    CREATE TEMP TABLE {arm}_messages AS
    WITH latest AS (
      SELECT pairs.target_id, pairs.cutoff_timestamp, pairs.affinity,
        history.elapsed_seconds, history.departure_timestamp
      FROM {arm}_pairs pairs ASOF LEFT JOIN feature_base history
        ON pairs.service_date = history.service_date
        AND pairs.source_station = history.parent_station
        AND pairs.trunk_route_id = history.trunk_route_id
        AND pairs.direction_id = history.direction_id
        AND pairs.cutoff_timestamp > history.departure_timestamp
    )
    SELECT target_id,
      sum(affinity * elapsed_seconds) / sum(affinity) FILTER (WHERE elapsed_seconds IS NOT NULL) AS message_seconds,
      sum(affinity * (cutoff_timestamp - departure_timestamp))
        / sum(affinity) FILTER (WHERE elapsed_seconds IS NOT NULL) AS age_seconds,
      count(elapsed_seconds) AS observed
    FROM latest GROUP BY target_id
    """
  )


@dataclass(frozen=True)
class Examples:
  arm_features: dict[str, Any]
  target: Any
  anchor: Any
  temporal: Any
  plan: Any
  plan_observed: Any
  persistence: Any
  persistence_observed: Any
  route: Any
  schedule_resolved: Any
  ambiguous_run_source: Any
  ambiguous_run_target: Any


def validation(connection: Any, feature_protocol: dict) -> dict:
  train = _examples(connection, "train", feature_protocol)
  held_out = _examples(connection, "validation", feature_protocol)
  baselines = _baselines(held_out)
  results = [
    _fit(arm, seed, train, held_out)
    for arm in ARMS
    for seed in SEEDS
  ]
  self_mae = sum(result["metrics"]["mae_seconds"] for result in results if result["arm"] == "self") / len(SEEDS)
  topology_gate = _claim_gate(results, baselines)
  return {
    "schema": 1,
    "split": "validation",
    "targets": len(held_out.target),
    "protocol_sha256": _digest(feature_protocol),
    "baselines": baselines,
    "results": results,
    "station_local_gate": {
      "mean_mae_seconds": round(self_mae, 6),
      "best_stage_3_mae_seconds": baselines["plan"]["mae_seconds"],
      "passed": self_mae < baselines["plan"]["mae_seconds"],
    },
    "topology_validation_gate": topology_gate,
    "decision": "freeze:open_learned_test_once" if topology_gate["passed"] else "stop:no_validation_topology_signal",
  }


def test(connection: Any, feature_protocol: dict, frozen_validation: dict) -> dict:
  if frozen_validation.get("decision") != "freeze:open_learned_test_once":
    raise ExperimentError("test: validation did not pass the frozen topology gate")
  held_out = _examples(connection, "test", feature_protocol)
  results = [
    _evaluate_frozen(result, held_out)
    for result in frozen_validation["results"]
  ]
  baselines = _baselines(held_out)
  gate = _claim_gate(results, baselines)
  return {
    "schema": 1,
    "split": "test",
    "targets": len(held_out.target),
    "protocol_sha256": _digest(feature_protocol),
    "validation_sha256": _digest(frozen_validation),
    "baselines": baselines,
    "results": results,
    "claim_gate": gate,
    "decision": "retain:topology_specific_signal" if gate["passed"] else "stop:no_topology_specific_signal",
  }


def _claim_gate(results: list[dict], baselines: dict) -> dict:
  controls = ("self", "reverse", "permuted")
  mean = {
    metric: {
      arm: sum(result["metrics"][metric] for result in results if result["arm"] == arm) / len(SEEDS)
      for arm in ARMS
    }
    for metric in ("mae_seconds", "macro_route_mae_seconds", "p90_absolute_error")
  }
  consistent = all(
    next(result for result in results if (result["arm"], result["seed"]) == ("true", seed))["metrics"]["mae_seconds"]
    < min(
      next(result for result in results if (result["arm"], result["seed"]) == (arm, seed))["metrics"]["mae_seconds"]
      for arm in controls
    )
    for seed in SEEDS
  )
  macro = mean["macro_route_mae_seconds"]["true"] < min(mean["macro_route_mae_seconds"][arm] for arm in controls)
  tail = mean["p90_absolute_error"]["true"] < min(mean["p90_absolute_error"][arm] for arm in controls)
  baselines_passed = mean["mae_seconds"]["true"] < min(
    baselines["temporal"]["mae_seconds"],
    baselines["plan"]["mae_seconds"],
  )
  positive = consistent and macro and tail and baselines_passed
  return {
    "mean": {
      metric: {arm: round(value, 6) for arm, value in arms.items()}
      for metric, arms in mean.items()
    },
    "true_beats_both_stage_3_baselines": baselines_passed,
    "true_beats_every_topology_control_in_every_seed": consistent,
    "true_beats_controls_on_macro_route_mae": macro,
    "true_beats_controls_on_p90_absolute_error": tail,
    "passed": positive,
  }


def _examples(connection: Any, split: str, feature_protocol: dict) -> Examples:
  import numpy as np

  values = connection.execute(
    """
    SELECT elapsed_seconds, anchor_seconds, temporal_seconds, plan_value, plan_observed,
      persistence_value, persistence_observed, weekday, local_hour, route_id,
      self_message_seconds, self_age_seconds, coalesce(self_observed, 0) AS self_observed,
      true_message_seconds, true_age_seconds, coalesce(true_observed, 0) AS true_observed,
      reverse_message_seconds, reverse_age_seconds, coalesce(reverse_observed, 0) AS reverse_observed,
      permuted_message_seconds, permuted_age_seconds, coalesce(permuted_observed, 0) AS permuted_observed,
      schedule_resolved, ambiguous_run_source, ambiguous_run_target
    FROM topology_features WHERE split = ? ORDER BY target_id
    """,
    [split],
  ).fetchnumpy()
  route_names = [row["route_id"] for row in feature_protocol["route_blend"]]
  persistence = values["persistence_value"].astype("float64")
  shared = (
    np.log1p(values["temporal_seconds"]),
    np.log1p(values["plan_value"]),
    np.log1p(persistence),
    values["plan_observed"],
    values["persistence_observed"],
    np.sin(2 * pi * values["weekday"] / 7),
    np.cos(2 * pi * values["weekday"] / 7),
    np.sin(2 * pi * values["local_hour"] / 24),
    np.cos(2 * pi * values["local_hour"] / 24),
  )
  routes = tuple((values["route_id"] == route).astype("float64") for route in route_names)
  arm_features = {}
  for arm in ARMS:
    message = np.ma.filled(values[f"{arm}_message_seconds"], persistence).astype("float64")
    age = np.ma.filled(values[f"{arm}_age_seconds"], 0).astype("float64")
    arm_features[arm] = np.stack(
      (*shared, np.log1p(message), np.log1p(age), values[f"{arm}_observed"] > 0, *routes),
      axis=1,
    ).astype("float32")
  return Examples(
    arm_features,
    values["elapsed_seconds"].astype("float32"),
    values["anchor_seconds"].astype("float32"),
    values["temporal_seconds"].astype("float32"),
    values["plan_value"].astype("float32"),
    values["plan_observed"].astype("bool"),
    persistence.astype("float32"),
    values["persistence_observed"].astype("bool"),
    values["route_id"],
    values["schedule_resolved"].astype("bool"),
    values["ambiguous_run_source"].astype("bool"),
    values["ambiguous_run_target"].astype("bool"),
  )


def _fit(arm: str, seed: int, train: Examples, validation: Examples) -> dict:
  import numpy as np

  train_features = train.arm_features[arm]
  validation_features = validation.arm_features[arm]
  mean = train_features.mean(axis=0)
  scale = train_features.std(axis=0)
  scale[scale < 1e-6] = 1
  transformed_target = np.log1p(train.target)
  target_mean = float(transformed_target.mean())
  target_scale = float(transformed_target.std())
  training = ((train_features - mean) / scale).astype("float32")
  held_out = ((validation_features - mean) / scale).astype("float32")
  normalized_target = ((transformed_target - target_mean) / target_scale).astype("float32")
  normalized_anchor = ((np.log1p(train.anchor) - target_mean) / target_scale).astype("float32")
  validation_anchor = ((np.log1p(validation.anchor) - target_mean) / target_scale).astype("float32")
  Tensor.manual_seed(seed)
  model = ResidualModel(training.shape[1], HIDDEN)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE, fused=False)
  random = np.random.default_rng(seed)
  initial = _predict(model, held_out, validation_anchor, target_mean, target_scale)
  best_mae = float(np.abs(initial - validation.target).mean())
  best_step, best_state = 0, _state(model)
  checkpoints = [{"step": 0, "mae_seconds": round(best_mae, 6)}]
  for step in range(1, STEPS + 1):
    index = random.integers(0, len(training), size=BATCH_SIZE)
    features = Tensor(training[index])
    target = Tensor(normalized_target[index])
    anchor = Tensor(normalized_anchor[index])
    with Context(TRAINING=1):
      optimizer.zero_grad()
      error = model(features, anchor) - target
      absolute = error.abs()
      loss = (absolute < HUBER_DELTA).where(
        error.square() / (2 * HUBER_DELTA),
        absolute - HUBER_DELTA / 2,
      ).mean().backward()
      loss.realize(*optimizer.schedule_step())
    if step % CHECKPOINT_EVERY:
      continue
    prediction = _predict(model, held_out, validation_anchor, target_mean, target_scale)
    mae = float(np.abs(prediction - validation.target).mean())
    checkpoints.append({"step": step, "mae_seconds": round(mae, 6)})
    if mae < best_mae:
      best_mae, best_step, best_state = mae, step, _state(model)
  nn.state.load_state_dict(model, _tensors(best_state), verbose=False)
  prediction = _predict(model, held_out, validation_anchor, target_mean, target_scale)
  scaler = {
    "feature_mean": mean.tolist(),
    "feature_scale": scale.tolist(),
    "target_mean": target_mean,
    "target_scale": target_scale,
  }
  return {
    "arm": arm,
    "seed": seed,
    "best_step": best_step,
    "parameters": sum(parameter.numel() for parameter in nn.state.get_parameters(model)),
    "trained_examples": STEPS * BATCH_SIZE,
    "checkpoints": checkpoints,
    "metrics": _metrics(validation, prediction),
    "route_metrics": _route_metrics(validation, prediction),
    "mask_metrics": _mask_metrics(validation, prediction),
    "scaler": scaler,
    "state": best_state,
  }


def _evaluate_frozen(frozen: dict, examples: Examples) -> dict:
  import numpy as np

  scaler = frozen["scaler"]
  features = ((examples.arm_features[frozen["arm"]] - np.asarray(scaler["feature_mean"])) / np.asarray(scaler["feature_scale"])).astype("float32")
  anchor = ((np.log1p(examples.anchor) - scaler["target_mean"]) / scaler["target_scale"]).astype("float32")
  model = ResidualModel(features.shape[1], HIDDEN)
  nn.state.load_state_dict(model, _tensors(frozen["state"]), verbose=False)
  prediction = _predict(model, features, anchor, scaler["target_mean"], scaler["target_scale"])
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


def _predict(model: ResidualModel, features: Any, anchor: Any, target_mean: float, target_scale: float) -> Any:
  import numpy as np

  chunks = []
  for start in range(0, len(features), EVALUATION_BATCH_SIZE):
    stop = start + EVALUATION_BATCH_SIZE
    chunks.append(model(Tensor(features[start:stop]), Tensor(anchor[start:stop])).numpy())
  return np.expm1(np.concatenate(chunks) * target_scale + target_mean)


def _state(model: ResidualModel) -> dict[str, dict]:
  return {
    name: {
      "shape": list(value.shape),
      "float32_base64": base64.b64encode(value.numpy().astype("<f4").tobytes()).decode(),
    }
    for name, value in nn.state.get_state_dict(model).items()
  }


def _tensors(state: dict[str, dict]) -> dict[str, Tensor]:
  import numpy as np

  return {
    name: Tensor(np.frombuffer(base64.b64decode(value["float32_base64"]), dtype="<f4").reshape(value["shape"]).copy())
    for name, value in state.items()
  }


def _baselines(examples: Examples) -> dict:
  return {
    "persistence": _metrics(examples, examples.persistence, examples.persistence_observed),
    "temporal": _metrics(examples, examples.temporal),
    "plan": _metrics(examples, examples.plan, examples.plan_observed),
    "anchor": _metrics(examples, examples.anchor),
  }


def _metrics(examples: Examples, prediction: Any, observed: Any | None = None) -> dict:
  import numpy as np

  mask = np.ones(len(examples.target), dtype="bool") if observed is None else observed
  error = prediction[mask] - examples.target[mask]
  absolute = np.abs(error)
  route_mae = [absolute[examples.route[mask] == route].mean() for route in sorted(set(examples.route[mask]))]
  return {
    "predictions": int(mask.sum()),
    "targets": len(examples.target),
    "coverage": round(float(mask.mean()), 6),
    "mae_seconds": round(float(absolute.mean()), 6),
    "rmse_seconds": round(float(sqrt(np.square(error).mean())), 6),
    "median_absolute_error": round(float(np.quantile(absolute, 0.5, method="lower")), 6),
    "p90_absolute_error": round(float(np.quantile(absolute, 0.9, method="lower")), 6),
    "macro_route_mae_seconds": round(float(np.mean(route_mae)), 6),
    "nonpositive_predictions": int((prediction[mask] <= 0).sum()),
  }


def _route_metrics(examples: Examples, prediction: Any) -> list[dict]:
  import numpy as np

  return [
    {
      "route_id": route,
      "targets": int(mask.sum()),
      "mae_seconds": round(float(np.abs(prediction[mask] - examples.target[mask]).mean()), 6),
    }
    for route in sorted(set(examples.route))
    if (mask := examples.route == route).any()
  ]


def _mask_metrics(examples: Examples, prediction: Any) -> list[dict]:
  import numpy as np

  masks = {
    "schedule_resolved": examples.schedule_resolved,
    "ambiguous_run_source": examples.ambiguous_run_source,
    "ambiguous_run_target": examples.ambiguous_run_target,
  }
  return [
    {
      "mask": name,
      "value": value,
      "targets": int(selected.sum()),
      "mae_seconds": round(float(np.abs(prediction[selected] - examples.target[selected]).mean()), 6) if selected.any() else None,
    }
    for name, mask in masks.items()
    for value in (False, True)
    if (selected := mask == value).any()
  ]


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
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--test", action="store_true")
  arguments = parser.parse_args()
  arguments.output_dir.mkdir(parents=True, exist_ok=True)
  protocol_path = arguments.output_dir / "protocol.json"
  validation_path = arguments.output_dir / "validation.json"
  test_path = arguments.output_dir / "test.json"
  if arguments.test and test_path.exists():
    raise ExperimentError("test.json: learned test split was already opened")
  frozen_protocol = _read(protocol_path) if arguments.test else None
  frozen_validation = _read(validation_path) if arguments.test else None
  connection, result = build(arguments.source_dir, arguments.population_audit, arguments.task_protocol)
  if frozen_protocol is not None and result != frozen_protocol:
    raise ExperimentError("test: frozen feature protocol does not match rebuilt facts")
  _freeze(protocol_path, result)
  if arguments.test:
    evidence = test(connection, result, frozen_validation)
    _write(test_path, evidence)
  else:
    evidence = validation(connection, result)
    _freeze(validation_path, evidence)
  print(json.dumps({"protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "split": evidence["split"]}))


if __name__ == "__main__":
  main()
