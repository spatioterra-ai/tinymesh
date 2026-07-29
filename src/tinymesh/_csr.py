from typing import cast

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo


class _CSR:
    def __init__(
        self,
        nodes: int,
        source: tuple[int, ...],
        target: tuple[int, ...],
    ) -> None:
        self.nodes = nodes
        self.source = source
        self.target = target
        self.column, self.row_ptr, self.edge_order = _group(nodes, target, source)
        self.transpose_column, self.transpose_row_ptr, self.transpose_order = _group(nodes, source, target)
        self._tensors_by_device: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        self._edge_tensors_by_device: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}

    def sum(self, values: Tensor) -> Tensor:
        if self.nodes == 1:
            return values * len(self.column)
        if not self.column:
            return values * 0
        device = cast(str, values.device)
        row_ptr, column, transpose_row_ptr, transpose_column = self._tensors(device)
        output = Tensor.invalids(*values.shape, dtype=values.dtype, device=device)
        return output.custom_kernel(
            values,
            row_ptr,
            column,
            transpose_row_ptr,
            transpose_column,
            fxn=_csr_sum_kernel,
            grad_fxn=_csr_sum_gradient,
        )[0]

    def weighted_sum(self, values: Tensor, edge_weight: Tensor) -> Tensor:
        if self.nodes == 1 or not self.column:
            return values * edge_weight.sum()
        device = cast(str, values.device)
        row_ptr, column, transpose_row_ptr, transpose_column = self._tensors(device)
        edge_order, transpose_order, source, target = self._edge_tensors(device)
        output = Tensor.invalids(*values.shape, dtype=values.dtype, device=device)
        return output.custom_kernel(
            values,
            edge_weight,
            row_ptr,
            column,
            edge_order,
            transpose_row_ptr,
            transpose_column,
            transpose_order,
            source,
            target,
            fxn=_csr_reindexed_weighted_sum_kernel,
            grad_fxn=_csr_weighted_gradient,
        )[0]

    def edge_values(self, values: Tensor, *, source: bool) -> Tensor:
        if not self.column:
            return values[:0]
        if self.nodes == 1:
            return values.expand(len(self.source), values.shape[1])
        device = cast(str, values.device)
        row_ptr, _, transpose_row_ptr, _ = self._tensors(device)
        edge_order, transpose_order, source_index, target_index = self._edge_tensors(device)
        index, grouped_row_ptr, grouped_edge = (
            (source_index, transpose_row_ptr, transpose_order)
            if source
            else (target_index, row_ptr, edge_order)
        )
        output = Tensor.invalids(len(self.source), values.shape[1], dtype=values.dtype, device=device)
        return output.custom_kernel(
            values,
            index,
            grouped_row_ptr,
            grouped_edge,
            fxn=_edge_values_kernel,
            grad_fxn=_edge_values_gradient,
        )[0]

    def softmax(self, edge_score: Tensor) -> Tensor:
        if not self.column:
            return edge_score
        score = edge_score.reshape(-1, 1)
        maximum = self._segment_max(score).detach()
        exponential = (score - self.edge_values(maximum, source=False)).exp()
        total = self._segment_sum(exponential)
        return (exponential / self.edge_values(total, source=False)).reshape(-1)

    def in_degree(self, device: str) -> Tensor:
        row_ptr = self._tensors(device)[0]
        return row_ptr[1:] - row_ptr[:-1]

    def _segment_sum(self, edge_values: Tensor) -> Tensor:
        if self.nodes == 1:
            return edge_values.sum(axis=0, keepdim=True)
        device = cast(str, edge_values.device)
        row_ptr, _, _, _ = self._tensors(device)
        edge_order, _, _, target = self._edge_tensors(device)
        output = Tensor.invalids(self.nodes, edge_values.shape[1], dtype=edge_values.dtype, device=device)
        return output.custom_kernel(
            edge_values,
            row_ptr,
            edge_order,
            target,
            fxn=_csr_sum_kernel,
            grad_fxn=_segment_sum_gradient,
        )[0]

    def _segment_max(self, edge_values: Tensor) -> Tensor:
        if self.nodes == 1:
            return edge_values.max(axis=0, keepdim=True)
        device = cast(str, edge_values.device)
        row_ptr, _, _, _ = self._tensors(device)
        edge_order, _, _, _ = self._edge_tensors(device)
        output = Tensor.invalids(self.nodes, edge_values.shape[1], dtype=edge_values.dtype, device=device)
        return output.custom_kernel(
            edge_values,
            row_ptr,
            edge_order,
            fxn=_segment_max_kernel,
        )[0]

    def _tensors(self, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = self._tensors_by_device.get(device)
        if tensors is None:
            tensors = (
                Tensor(self.row_ptr, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.column, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_row_ptr, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_column, dtype=dtypes.int32, device=device).realize(),
            )
            self._tensors_by_device[device] = tensors
        return tensors

    def _edge_tensors(self, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = self._edge_tensors_by_device.get(device)
        if tensors is None:
            tensors = (
                Tensor(self.edge_order, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.transpose_order, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.source, dtype=dtypes.int32, device=device).realize(),
                Tensor(self.target, dtype=dtypes.int32, device=device).realize(),
            )
            self._edge_tensors_by_device[device] = tensors
        return tensors


def _group(
    nodes: int,
    owner: tuple[int, ...],
    neighbor: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows: list[list[tuple[int, int]]] = [[] for _ in range(nodes)]
    for edge, (row_index, column) in enumerate(zip(owner, neighbor)):
        rows[row_index].append((column, edge))
    flat = tuple(item for row in rows for item in sorted(row))
    row_ptr = [0]
    for row in rows:
        row_ptr.append(row_ptr[-1] + len(row))
    return (
        tuple(column for column, _ in flat),
        tuple(row_ptr),
        tuple(edge for _, edge in flat),
    )


def _csr_sum_kernel(output: UOp, values: UOp, row_ptr: UOp, column: UOp, *_: UOp) -> UOp:
    return _csr_kernel(output, values, row_ptr, column)


def _csr_reindexed_weighted_sum_kernel(
    output: UOp,
    values: UOp,
    edge_weight: UOp,
    row_ptr: UOp,
    column: UOp,
    weight_index: UOp,
    *_: UOp,
) -> UOp:
    return _csr_kernel(output, values, row_ptr, column, edge_weight, weight_index)


def _csr_kernel(
    output: UOp,
    values: UOp,
    row_ptr: UOp,
    column: UOp,
    edge_weight: UOp | None = None,
    weight_index: UOp | None = None,
) -> UOp:
    nodes, width = output.shape
    output, values, row_ptr, column = output.base, values.base, row_ptr.base, column.base
    if edge_weight is not None:
        edge_weight = edge_weight.base
    if weight_index is not None:
        weight_index = weight_index.base
    lane = UOp.range(nodes * width, 0, dtype=dtypes.int32)
    row, feature = lane // width, lane % width
    start, stop = row_ptr[row].cast(dtypes.int32), row_ptr[row + 1].cast(dtypes.int32)

    accumulator = UOp.placeholder((1,), values.dtype, 0, addrspace=AddrSpace.REG)
    edge = UOp.placeholder((1,), dtypes.int32, 1, addrspace=AddrSpace.REG)
    accumulator_init = accumulator.after(lane)[0].store(0.0)
    accumulator = accumulator.after(accumulator_init)
    edge_init = edge.after(accumulator_init)[0].store(start)
    edge = edge.after(edge_init)
    loop = UOp.loop(1)

    current = edge.after(loop)[0].load()
    active = current < stop
    position = active.where(current, 0)
    source = column[position].load().cast(dtypes.int32)
    message = values[source * width + feature]
    if edge_weight is not None:
        edge_index = position if weight_index is None else weight_index[position].load().cast(dtypes.int32)
        message = message * edge_weight[edge_index]
    next_edge = current + 1
    updated = UOp.group(
        accumulator[0].store(
            accumulator.after(loop)[0].load()
            + active.where(message, 0.0)
        ),
        edge[0].store(next_edge),
    )
    done = updated.end(loop, next_edge < stop)
    return output[lane].store(accumulator.after(done)[0].load()).end(lane).sink(
        arg=KernelInfo(name="csr_sum", opts_to_apply=())
    )


def _edge_dot_kernel(
    output: UOp,
    values: UOp,
    gradient: UOp,
    source: UOp,
    target: UOp,
) -> UOp:
    edges, width = output.shape[0], values.shape[1]
    edge = UOp.range(edges, 0, dtype=dtypes.int32)
    feature = UOp.range(width, 1, axis_type=AxisType.REDUCE)
    edge_source = source[edge].load().cast(dtypes.int32)
    edge_target = target[edge].load().cast(dtypes.int32)
    output = output[edge].set(0.0)
    output = output[edge].set(
        output.after(feature)[edge] + values[edge_source, feature] * gradient[edge_target, feature],
        end=feature,
    )
    return output.end(edge).sink(arg=KernelInfo(name="edge_dot", opts_to_apply=()))


def _edge_values_kernel(output: UOp, values: UOp, index: UOp, *_: UOp) -> UOp:
    edges, width = output.shape
    output, values, index = output.base, values.base, index.base
    lane = UOp.range(edges * width, 0, dtype=dtypes.int32)
    edge, feature = lane // width, lane % width
    node = index[edge].load().cast(dtypes.int32)
    return output[lane].store(values[node * width + feature]).end(lane).sink(
        arg=KernelInfo(name="edge_values")
    )


def _segment_max_kernel(output: UOp, values: UOp, row_ptr: UOp, edge_order: UOp) -> UOp:
    nodes, width = output.shape
    output, values, row_ptr, edge_order = output.base, values.base, row_ptr.base, edge_order.base
    lane = UOp.range(nodes * width, 0, dtype=dtypes.int32)
    row, feature = lane // width, lane % width
    start, stop = row_ptr[row].cast(dtypes.int32), row_ptr[row + 1].cast(dtypes.int32)

    maximum = UOp.placeholder((1,), values.dtype, 0, addrspace=AddrSpace.REG)
    edge = UOp.placeholder((1,), dtypes.int32, 1, addrspace=AddrSpace.REG)
    maximum_init = maximum.after(lane)[0].store(float("-inf"))
    maximum = maximum.after(maximum_init)
    edge_init = edge.after(maximum_init)[0].store(start)
    edge = edge.after(edge_init)
    loop = UOp.loop(1)

    current = edge.after(loop)[0].load()
    active = current < stop
    position = active.where(current, 0)
    edge_index = edge_order[position].load().cast(dtypes.int32)
    value = values[edge_index * width + feature]
    next_edge = current + 1
    updated = UOp.group(
        maximum[0].store(
            maximum.after(loop)[0].load().maximum(active.where(value, float("-inf")))
        ),
        edge[0].store(next_edge),
    )
    done = updated.end(loop, next_edge < stop)
    result = (start < stop).where(maximum.after(done)[0].load(), 0.0)
    return output[lane].store(result).end(lane).sink(
        arg=KernelInfo(name="segment_max", opts_to_apply=())
    )


def _csr_sum_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    _, values, _, _, transpose_row_ptr, transpose_column = call.src[1:]
    output = Tensor.invalids(*values.shape, dtype=values.dtype, device=values.device)
    grad_values = output.custom_kernel(
        Tensor(gradient),
        Tensor(transpose_row_ptr),
        Tensor(transpose_column),
        fxn=_csr_sum_kernel,
    )[0]
    return None, grad_values.uop, None, None, None, None


def _csr_weighted_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    (
        _,
        values,
        edge_weight,
        _,
        _,
        _,
        transpose_row_ptr,
        transpose_column,
        transpose_order,
        source,
        target,
    ) = call.src[1:]
    values_output = Tensor.invalids(*values.shape, dtype=values.dtype, device=values.device)
    grad_values = values_output.custom_kernel(
        Tensor(gradient),
        Tensor(edge_weight),
        Tensor(transpose_row_ptr),
        Tensor(transpose_column),
        Tensor(transpose_order),
        fxn=_csr_reindexed_weighted_sum_kernel,
    )[0]
    weight_output = Tensor.invalids(*edge_weight.shape, dtype=edge_weight.dtype, device=edge_weight.device)
    grad_weight = weight_output.custom_kernel(
        Tensor(values),
        Tensor(gradient),
        Tensor(source),
        Tensor(target),
        fxn=_edge_dot_kernel,
    )[0]
    return None, grad_values.uop, grad_weight.uop, None, None, None, None, None, None, None, None


def _edge_values_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    _, values, _, row_ptr, edge_order = call.src[1:]
    output = Tensor.invalids(*values.shape, dtype=values.dtype, device=values.device)
    grad_values = output.custom_kernel(
        Tensor(gradient),
        Tensor(row_ptr),
        Tensor(edge_order),
        fxn=_csr_sum_kernel,
    )[0]
    return None, grad_values.uop, None, None, None


def _segment_sum_gradient(gradient: UOp, call: UOp) -> tuple[UOp | None, ...]:
    _, edge_values, _, _, target = call.src[1:]
    output = Tensor.invalids(*edge_values.shape, dtype=edge_values.dtype, device=edge_values.device)
    grad_edge_values = output.custom_kernel(
        Tensor(gradient),
        Tensor(target),
        fxn=_edge_values_kernel,
    )[0]
    return None, grad_edge_values.uop, None, None, None
