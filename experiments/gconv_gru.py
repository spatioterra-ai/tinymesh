"""Compare graph-convolutional recurrence with T-GCN."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn
from tinygrad.uop.ops import Ops

from experiments.tgcn import TGCN
from tinymesh import Graph


class ChebConv:
    def __init__(self, in_features: int, out_features: int, order: int) -> None:
        if in_features <= 0 or out_features <= 0 or order <= 0:
            raise ValueError("feature counts and order must be positive")
        self.linear = nn.Linear(order * in_features, out_features)
        self.in_features = in_features
        self.order = order

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        _validate_graph(graph)
        return self._project(values, graph)

    def _project(self, values: Tensor, graph: Graph) -> Tensor:
        expected = (graph.nodes, self.in_features)
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}, got {values.shape}")
        if not isinstance(values.device, str):
            raise ValueError("ChebConv requires one device")

        degree = graph.in_degree(device=values.device)
        scale = (degree != 0).where(
            degree.maximum(1).cast(values.dtype).rsqrt(),
            0,
        ).reshape(-1, 1)
        states = [values]
        if self.order > 1:
            states.append(_shift(values, graph, scale))
        for _ in range(2, self.order):
            states.append(2 * _shift(states[-1], graph, scale) - states[-2])
        basis = states[0] if self.order == 1 else states[0].cat(*states[1:], dim=1)
        return self.linear(basis)


class GConvGRU:
    def __init__(self, in_features: int, hidden_features: int, order: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        self.gates = ChebConv(in_features + hidden_features, 2 * hidden_features, order)
        self.candidate = ChebConv(in_features + hidden_features, hidden_features, order)
        self.in_features = in_features
        self.hidden_features = hidden_features

    def __call__(self, values: Tensor, graph: Graph, hidden: Tensor | None = None) -> Tensor:
        _validate_graph(graph)
        expected = (graph.nodes, self.in_features)
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}, got {values.shape}")
        if not isinstance(values.device, str):
            raise ValueError("GConvGRU requires one device")
        hidden = self._hidden(values, hidden)

        gates = self.gates._project(values.cat(hidden, dim=1), graph)
        update = gates[:, :self.hidden_features].sigmoid()
        reset = gates[:, self.hidden_features:].sigmoid()
        candidate = self.candidate._project(values.cat(hidden * reset, dim=1), graph).tanh()
        return update * hidden + (1 - update) * candidate

    def _hidden(self, values: Tensor, hidden: Tensor | None) -> Tensor:
        if hidden is None:
            return Tensor.zeros(
                values.shape[0],
                self.hidden_features,
                dtype=values.dtype,
                device=values.device,
            )
        expected = (values.shape[0], self.hidden_features)
        if hidden.shape != expected:
            raise ValueError(f"hidden must have shape {expected}, got {hidden.shape}")
        if hidden.dtype != values.dtype or hidden.device != values.device:
            raise ValueError("hidden and values must share dtype and device")
        return hidden


@dataclass(frozen=True)
class Observation:
    device: str
    steps: int
    tgcn_parameters: int
    gconv_gru_parameters: int
    tgcn_sparse_calls: int
    gconv_gru_sparse_calls: int
    tgcn_initial_loss: float
    gconv_gru_initial_loss: float
    tgcn_spatial_gradient: float
    gconv_gru_spatial_gradient: float
    tgcn_final_loss: float
    gconv_gru_final_loss: float


def compare(device: str) -> Observation:
    graph = Graph(2, [0, 1], [1, 0])
    snapshots = (
        Tensor([[1.0], [0.0]], device=device).realize(),
        Tensor.zeros(2, 1, device=device).realize(),
    )
    target = Tensor([[1.0]], device=device).realize()
    tgcn, gconv_gru = TGCN(1, 1), GConvGRU(1, 1, 2)
    _set_tgcn_parameters(tgcn, device)
    _set_gconv_gru_parameters(gconv_gru, device)

    tgcn_initial, tgcn_gradient, tgcn_final = _fit(
        tgcn,
        tgcn.graph_projection.linear.weight,
        (2, 0),
        graph,
        snapshots,
        target,
    )
    gconv_initial, gconv_gradient, gconv_final = _fit(
        gconv_gru,
        gconv_gru.candidate.linear.weight,
        (0, 2),
        graph,
        snapshots,
        target,
    )
    return Observation(
        device,
        steps=1,
        tgcn_parameters=_parameter_count(tgcn),
        gconv_gru_parameters=_parameter_count(gconv_gru),
        tgcn_sparse_calls=_sparse_calls(tgcn(snapshots[0], graph)),
        gconv_gru_sparse_calls=_sparse_calls(gconv_gru(snapshots[0], graph)),
        tgcn_initial_loss=tgcn_initial,
        gconv_gru_initial_loss=gconv_initial,
        tgcn_spatial_gradient=tgcn_gradient,
        gconv_gru_spatial_gradient=gconv_gradient,
        tgcn_final_loss=tgcn_final,
        gconv_gru_final_loss=gconv_final,
    )


def _shift(values: Tensor, graph: Graph, scale: Tensor) -> Tensor:
    return -graph.sum(values * scale) * scale


def _validate_graph(graph: Graph) -> None:
    if any(source == target for source, target in zip(graph.source, graph.target)):
        raise ValueError("Chebyshev graphs must not contain self-loops")
    edges = Counter(zip(graph.source, graph.target))
    reverse = Counter((target, source) for source, target in zip(graph.source, graph.target))
    if edges != reverse:
        raise ValueError("Chebyshev graphs must be symmetric")


def _fit(
    model: TGCN | GConvGRU,
    parameter: Tensor,
    parameter_index: tuple[int, ...],
    graph: Graph,
    snapshots: tuple[Tensor, ...],
    target: Tensor,
) -> tuple[float, float, float]:
    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=1.0, fused=False)

    def loss() -> Tensor:
        hidden = model(snapshots[0], graph)
        for snapshot in snapshots[1:]:
            hidden = model(snapshot, graph, hidden)
        return (hidden[1:] - target).square().mean()

    initial_loss = loss().item()
    with Context(TRAINING=1):
        optimizer.zero_grad()
        training_loss = loss().backward()
        if parameter.grad is None:
            raise RuntimeError("spatial parameter has no gradient")
        spatial_gradient = parameter.grad[parameter_index].item()
        training_loss.realize(*optimizer.schedule_step())
    return initial_loss, spatial_gradient, loss().item()


def _set_tgcn_parameters(model: TGCN, device: str) -> None:
    model.graph_projection.linear.weight = Tensor([[0.0], [0.0], [1.0]], device=device).realize()
    for gate in (model.update, model.reset):
        gate.weight = Tensor.zeros(1, 2, device=device).realize()
        gate.bias = Tensor.zeros(1, device=device).realize()
    model.candidate.weight = Tensor([[1.0, 0.0]], device=device).realize()
    model.candidate.bias = Tensor.zeros(1, device=device).realize()


def _set_gconv_gru_parameters(model: GConvGRU, device: str) -> None:
    model.gates.linear.weight = Tensor.zeros(2, 4, device=device).realize()
    model.gates.linear.bias = Tensor.zeros(2, device=device).realize()
    model.candidate.linear.weight = Tensor([[0.0, 0.0, -1.0, 0.0]], device=device).realize()
    model.candidate.linear.bias = Tensor.zeros(1, device=device).realize()


def _parameter_count(model: TGCN | GConvGRU) -> int:
    return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _sparse_calls(output: Tensor) -> int:
    return sum(
        uop.src[0].arg.name == "csr_sum"
        for uop in output.uop.toposort()
        if uop.op is Ops.CALL
    )


def main() -> None:
    print(json.dumps(asdict(compare(Device.DEFAULT)), indent=2))


if __name__ == "__main__":
    main()
