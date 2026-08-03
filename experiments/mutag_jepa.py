"""Test a masked whole-graph JEPA representation on MUTAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.helpers import getenv

from experiments.mutag_protocol import Metric, Probe, linear_probe, metric, molecular_summary, stratified_folds
from tinymesh import Graph
from tinymesh.datasets import MUTAG, mutag
from tinymesh.nn import SAGEConv


class GraphBatch:
    """Sparse disjoint union plus sparse node-to-graph pooling."""

    def __init__(self, data: MUTAG, indices: tuple[int, ...], *, mask_every: int, seed: int) -> None:
        source, target, node_labels, masked, pool_target = [], [], [], [], []
        offset = 0
        for row, index in enumerate(indices):
            graph, labels, _, _ = data[index]
            source.extend(offset + node for node in graph.source)
            target.extend(offset + node for node in graph.target)
            node_labels.extend(labels.tolist())
            masked.extend((node + index + seed) % mask_every == 0 for node in range(graph.nodes))
            pool_target.extend([row] * graph.nodes)
            offset += graph.nodes

        device = str(data.node_labels[0].device)
        atoms = Tensor(node_labels, device=device).one_hot(len(data.node_types)).float()
        mask = Tensor(masked, device=device).reshape(-1, 1)
        self.context = mask.where(0, atoms).cat(mask.float(), dim=1).clone().realize()
        self.target = atoms.cat(Tensor.zeros_like(mask).float(), dim=1).clone().realize()
        self.graph = Graph(offset, source, target)
        self.pool_graph = Graph(offset + len(indices), list(range(offset)), [offset + row for row in pool_target])
        self.graphs = len(indices)

    def pool(self, values: Tensor) -> Tensor:
        zeros = Tensor.zeros(self.graphs, values.shape[1], dtype=values.dtype, device=values.device)
        return self.pool_graph.mean(values.cat(zeros, dim=0))[self.graph.nodes:]


class Encoder:
    def __init__(self, in_features: int, hidden_features: int) -> None:
        self.first = SAGEConv(in_features, hidden_features)
        self.second = SAGEConv(hidden_features, hidden_features)
        self.projection = nn.Linear(hidden_features, hidden_features)

    def __call__(self, values: Tensor, batch: GraphBatch) -> Tensor:
        state = self.first(values, batch.graph).relu()
        state = self.second(state, batch.graph).relu()
        return self.projection(batch.pool(state)).tanh()


class Predictor:
    def __init__(self, hidden_features: int) -> None:
        self.hidden = nn.Linear(hidden_features, hidden_features // 2)
        self.output = nn.Linear(hidden_features // 2, hidden_features)

    def __call__(self, values: Tensor) -> Tensor:
        return self.output(self.hidden(values).relu())


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_graphs: int
    test_graphs: int
    initial_loss: float
    final_loss: float
    initial_target_sample_std: float
    target_sample_std: float
    target_parameter_delta: float
    target_gradient: float
    majority_accuracy: float
    summary: Probe
    random_encoder: Probe
    jepa_encoder: Probe


@dataclass(frozen=True)
class Observation:
    device: str
    seed: int
    folds: int
    steps: int
    probe_steps: int
    mask_every: int
    hidden_features: int
    learning_rate: float
    ema_decay: float
    probe_learning_rate: float
    parameters: int
    initial_loss: Metric
    final_loss: Metric
    initial_target_sample_std: Metric
    target_sample_std: Metric
    target_parameter_delta: Metric
    target_gradient: Metric
    majority_accuracy: Metric
    summary_accuracy: Metric
    random_encoder_accuracy: Metric
    jepa_encoder_accuracy: Metric
    results: tuple[FoldResult, ...]


def compare(
    data: MUTAG,
    *,
    seed: int,
    folds: int,
    steps: int,
    probe_steps: int,
    mask_every: int,
    hidden_features: int,
    learning_rate: float,
    ema_decay: float,
    probe_learning_rate: float,
) -> Observation:
    if steps <= 0 or probe_steps <= 0 or hidden_features < 2 or mask_every < 2:
        raise ValueError("steps and probe steps must be positive, hidden features >= 2, and mask interval >= 2")
    if learning_rate <= 0 or probe_learning_rate <= 0 or not 0 <= ema_decay < 1:
        raise ValueError("learning rates must be positive and EMA decay must be in [0, 1)")

    partitions = stratified_folds(data.labels, folds, seed)
    indices = tuple(range(len(data)))
    all_graphs = GraphBatch(data, indices, mask_every=mask_every, seed=0)
    summary = molecular_summary(data)
    results = []
    parameters = 0
    for fold, test in enumerate(partitions):
        test_set = set(test)
        train = tuple(index for index in indices if index not in test_set)
        result, parameters = _run_fold(
            data,
            all_graphs,
            summary,
            train,
            test,
            fold=fold,
            model_seed=seed * folds + fold,
            steps=steps,
            probe_steps=probe_steps,
            mask_every=mask_every,
            hidden_features=hidden_features,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            probe_learning_rate=probe_learning_rate,
        )
        results.append(result)

    results = tuple(results)
    return Observation(
        device=str(data.node_labels[0].device),
        seed=seed,
        folds=folds,
        steps=steps,
        probe_steps=probe_steps,
        mask_every=mask_every,
        hidden_features=hidden_features,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
        probe_learning_rate=probe_learning_rate,
        parameters=parameters,
        initial_loss=metric(tuple(result.initial_loss for result in results)),
        final_loss=metric(tuple(result.final_loss for result in results)),
        initial_target_sample_std=metric(tuple(result.initial_target_sample_std for result in results)),
        target_sample_std=metric(tuple(result.target_sample_std for result in results)),
        target_parameter_delta=metric(tuple(result.target_parameter_delta for result in results)),
        target_gradient=metric(tuple(result.target_gradient for result in results)),
        majority_accuracy=metric(tuple(result.majority_accuracy for result in results)),
        summary_accuracy=metric(tuple(result.summary.test_accuracy for result in results)),
        random_encoder_accuracy=metric(tuple(result.random_encoder.test_accuracy for result in results)),
        jepa_encoder_accuracy=metric(tuple(result.jepa_encoder.test_accuracy for result in results)),
        results=results,
    )


def _run_fold(
    data: MUTAG,
    all_graphs: GraphBatch,
    summary: Tensor,
    train: tuple[int, ...],
    test: tuple[int, ...],
    *,
    fold: int,
    model_seed: int,
    steps: int,
    probe_steps: int,
    mask_every: int,
    hidden_features: int,
    learning_rate: float,
    ema_decay: float,
    probe_learning_rate: float,
) -> tuple[FoldResult, int]:
    batch = GraphBatch(data, train, mask_every=mask_every, seed=model_seed)
    Tensor.manual_seed(model_seed)
    online = Encoder(batch.context.shape[1], hidden_features)
    target = Encoder(batch.context.shape[1], hidden_features)
    predictor = Predictor(hidden_features)
    _update_target(online, target, 0)
    initial_target = {
        name: value.detach().clone().realize()
        for name, value in nn.state.get_state_dict(target).items()
    }
    random_embedding = target(all_graphs.target, all_graphs).detach().clone().realize()
    optimizer = nn.optim.Adam(nn.state.get_parameters((online, predictor)), lr=learning_rate, fused=False)

    def loss() -> Tensor:
        truth = target(batch.target, batch).detach()
        prediction = predictor(online(batch.context, batch))
        return (prediction - truth).square().mean()

    initial_loss = loss().item()

    @TinyJit
    @Context(TRAINING=1)
    def step(context: Tensor, target_values: Tensor) -> Tensor:
        optimizer.zero_grad()
        truth = target(target_values, batch).detach()
        prediction = predictor(online(context, batch))
        value = (prediction - truth).square().mean().backward()
        return value.realize(*optimizer.schedule_step())

    for _ in range(steps):
        step(batch.context, batch.target)
        _update_target(online, target, ema_decay)

    trained_embedding = target(all_graphs.target, all_graphs).detach().clone().realize()
    probe_seed = model_seed + 10_000
    majority = max(range(2), key=lambda label: sum(data.labels[index] == label for index in train))
    result = FoldResult(
        fold=fold,
        train_graphs=len(train),
        test_graphs=len(test),
        initial_loss=initial_loss,
        final_loss=loss().item(),
        initial_target_sample_std=random_embedding.std(axis=0).mean().item(),
        target_sample_std=trained_embedding.std(axis=0).mean().item(),
        target_parameter_delta=sum(
            (value - initial_target[name]).abs().sum().item()
            for name, value in nn.state.get_state_dict(target).items()
        ),
        target_gradient=sum(
            0 if value.grad is None else value.grad.abs().sum().item()
            for value in nn.state.get_parameters(target)
        ),
        majority_accuracy=sum(data.labels[index] == majority for index in test) / len(test),
        summary=linear_probe(summary, data.labels, train, test, steps=probe_steps, learning_rate=probe_learning_rate, seed=probe_seed),
        random_encoder=linear_probe(
            random_embedding,
            data.labels,
            train,
            test,
            steps=probe_steps,
            learning_rate=probe_learning_rate,
            seed=probe_seed,
        ),
        jepa_encoder=linear_probe(
            trained_embedding,
            data.labels,
            train,
            test,
            steps=probe_steps,
            learning_rate=probe_learning_rate,
            seed=probe_seed,
        ),
    )
    parameters = sum(parameter.numel() for parameter in nn.state.get_parameters((online, predictor)))
    return result, parameters


def _update_target(online: Encoder, target: Encoder, decay: float) -> None:
    source = nn.state.get_state_dict(online)
    for name, value in nn.state.get_state_dict(target).items():
        value.assign(decay * value + (1 - decay) * source[name].detach()).realize()


def main() -> None:
    observation = compare(
        mutag(device=Device.DEFAULT),
        seed=getenv("SEED", 0),
        folds=getenv("FOLDS", 5),
        steps=getenv("STEPS", 100),
        probe_steps=getenv("PROBE_STEPS", 150),
        mask_every=getenv("MASK_EVERY", 3),
        hidden_features=getenv("HIDDEN", 16),
        learning_rate=getenv("LR", 0.01),
        ema_decay=getenv("EMA", 0.99),
        probe_learning_rate=getenv("PROBE_LR", 0.05),
    )
    print(json.dumps(asdict(observation), indent=2))


if __name__ == "__main__":
    main()
