from dataclasses import dataclass, field

from tinygrad import Tensor

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
        if values.ndim != 2:
            raise ValueError(f"values must have shape [N, H], got {values.shape}")
        if values.shape[0] != self.nodes:
            raise ValueError(f"values must have {self.nodes} rows, got {values.shape[0]}")
        if not isinstance(values.device, str):
            raise ValueError("graph sum requires one device")
        if edge_weight is None:
            return self._csr.sum(values)
        if edge_weight.ndim != 1 or edge_weight.shape[0] != self.edges:
            raise ValueError(f"edge_weight must have shape [{self.edges}], got {edge_weight.shape}")
        if values.dtype != edge_weight.dtype:
            raise ValueError(f"values and edge_weight must have the same dtype, got {values.dtype} and {edge_weight.dtype}")
        if edge_weight.device != values.device:
            raise ValueError("weighted graph sum requires one shared device")
        return self._csr.weighted_sum(values, edge_weight)

    def in_degree(self, *, device: str) -> Tensor:
        """Return incoming degree on one device."""
        if not isinstance(device, str):
            raise ValueError("in_degree requires one device")
        return self._csr.in_degree(device)
