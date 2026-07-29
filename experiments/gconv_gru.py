"""Compare graph-convolutional recurrence with T-GCN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, nn
from tinygrad.uop.ops import Ops

from tinymesh import Graph
from tinymesh.nn import GConvGRU, TGCN


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
