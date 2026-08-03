from dataclasses import dataclass, field
from math import prod
from typing import Literal

from tinygrad import Tensor, dtypes

from tinymesh._csr import _CSR


@dataclass(frozen=True, init=False)
class Graph:
    """An immutable directed graph over ordinary tinygrad tensors."""

    nodes: int
    source: tuple[int, ...]
    target: tuple[int, ...]
    _csr: _CSR = field(repr=False, compare=False)

    def __init__(
        self,
        nodes: int,
        source: list[int] | tuple[int, ...],
        target: list[int] | tuple[int, ...],
    ) -> None:
        source, target = tuple(source), tuple(target)
        if nodes <= 0:
            raise ValueError("nodes must be positive")
        if len(source) != len(target):
            raise ValueError("source and target must have the same length")
        if any(node < 0 or node >= nodes for edge in (source, target) for node in edge):
            raise ValueError(f"node IDs must be in [0, {nodes})")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "_csr", _CSR(nodes, source, target))

    @property
    def edges(self) -> int:
        return len(self.source)

    def sum(self, values: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        """Sum incoming values over node axis -2 with optional shared edge weights."""
        self._validate_node_values(values)
        if edge_weight is not None:
            if edge_weight.ndim != 1 or edge_weight.shape[0] != self.edges:
                raise ValueError(f"edge_weight must have shape [{self.edges}], got {edge_weight.shape}")
            if values.dtype != edge_weight.dtype:
                raise ValueError(f"values and edge_weight must have the same dtype, got {values.dtype} and {edge_weight.dtype}")
            if edge_weight.device != values.device:
                raise ValueError("weighted graph sum requires one shared device")

        shape = values.shape
        order = (values.ndim - 2, *range(values.ndim - 2), values.ndim - 1)
        flat = values.permute(order).reshape(self.nodes, -1)
        output = self._csr.sum(flat) if edge_weight is None else self._csr.weighted_sum(flat, edge_weight)
        return output.reshape(self.nodes, *shape[:-2], shape[-1]).permute(
            *range(1, values.ndim - 1),
            0,
            values.ndim - 1,
        )

    def mean(self, values: Tensor) -> Tensor:
        """Mean incoming values over node axis -2, with zero for empty rows."""
        self._validate_node_values(values)
        if not dtypes.is_float(values.dtype):
            raise ValueError(f"mean values must have a floating dtype, got {values.dtype}")
        assert isinstance(values.device, str)
        degree = self.in_degree(device=values.device).maximum(1).cast(values.dtype)
        degree = degree.reshape((1,) * (values.ndim - 2) + (self.nodes, 1))
        return self.sum(values) / degree

    def sum_edges(self, values: Tensor) -> Tensor:
        """Sum COO-ordered edge values at their target nodes over axis -2."""
        if values.ndim < 2:
            raise ValueError(f"values must have shape [..., E, H], got {values.shape}")
        if values.shape[-2] != self.edges:
            raise ValueError(f"values must have {self.edges} edge rows, got {values.shape[-2]}")
        if not isinstance(values.device, str):
            raise ValueError("edge values require one device")

        shape = values.shape
        order = (values.ndim - 2, *range(values.ndim - 2), values.ndim - 1)
        flat = values.permute(order).reshape(self.edges, prod(shape[:-2]) * shape[-1])
        output = self._csr._segment_sum(flat)
        return output.reshape(self.nodes, *shape[:-2], shape[-1]).permute(
            *range(1, values.ndim - 1),
            0,
            values.ndim - 1,
        )

    def edge_values(
        self,
        values: Tensor,
        *,
        endpoint: Literal["source", "target"],
    ) -> Tensor:
        """Gather node values into original COO edge order."""
        if values.ndim != 2:
            raise ValueError(f"values must have shape [N, H], got {values.shape}")
        self._validate_node_values(values)
        if endpoint not in ("source", "target"):
            raise ValueError("endpoint must be 'source' or 'target'")
        return self._csr.edge_values(values, source=endpoint == "source")

    def softmax(self, edge_score: Tensor) -> Tensor:
        """Normalize scalar edge scores over each target's incoming edges."""
        if edge_score.ndim != 1 or edge_score.shape[0] != self.edges:
            raise ValueError(f"edge_score must have shape [{self.edges}], got {edge_score.shape}")
        if not dtypes.is_float(edge_score.dtype):
            raise ValueError(f"edge_score must have a floating dtype, got {edge_score.dtype}")
        if not isinstance(edge_score.device, str):
            raise ValueError("graph softmax requires one device")
        return self._csr.softmax(edge_score)

    def in_degree(self, *, device: str) -> Tensor:
        """Return incoming degree on one device."""
        if not isinstance(device, str):
            raise ValueError("in_degree requires one device")
        return self._csr.in_degree(device)

    def _validate_node_values(self, values: Tensor) -> None:
        if values.ndim < 2:
            raise ValueError(f"values must have shape [..., N, H], got {values.shape}")
        if values.shape[-2] != self.nodes:
            raise ValueError(f"values must have {self.nodes} node rows, got {values.shape[-2]}")
        if not isinstance(values.device, str):
            raise ValueError("graph values require one device")
