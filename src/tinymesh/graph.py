from dataclasses import dataclass, field
from typing import Literal

from tinygrad import Tensor, dtypes

from tinymesh._csr import _CSR


@dataclass(frozen=True, init=False)
class Graph:
    """An immutable directed graph over ordinary Tinygrad tensors."""

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
        """Sum incoming values, optionally weighted in COO edge order."""
        self._validate_node_values(values)
        if edge_weight is None:
            return self._csr.sum(values)
        if edge_weight.ndim != 1 or edge_weight.shape[0] != self.edges:
            raise ValueError(f"edge_weight must have shape [{self.edges}], got {edge_weight.shape}")
        if values.dtype != edge_weight.dtype:
            raise ValueError(f"values and edge_weight must have the same dtype, got {values.dtype} and {edge_weight.dtype}")
        if edge_weight.device != values.device:
            raise ValueError("weighted graph sum requires one shared device")
        return self._csr.weighted_sum(values, edge_weight)

    def edge_values(
        self,
        values: Tensor,
        *,
        endpoint: Literal["source", "target"],
    ) -> Tensor:
        """Gather node values into original COO edge order."""
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
        if values.ndim != 2:
            raise ValueError(f"values must have shape [N, H], got {values.shape}")
        if values.shape[0] != self.nodes:
            raise ValueError(f"values must have {self.nodes} rows, got {values.shape[0]}")
        if not isinstance(values.device, str):
            raise ValueError("graph values require one device")
