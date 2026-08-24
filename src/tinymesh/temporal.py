from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from tinygrad import Tensor, dtypes

from tinymesh.graph import Graph


@dataclass(frozen=True)
class TemporalEdges:
    """Timestamped directed edges over one stable node universe."""

    nodes: int
    source: tuple[int, ...]
    target: tuple[int, ...]
    timestamp: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("source", "target", "timestamp"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.nodes, int) or isinstance(self.nodes, bool) or self.nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        if len({len(self.source), len(self.target), len(self.timestamp)}) != 1:
            raise ValueError("source, target, and timestamp must have the same length")
        for name, endpoints in (("source", self.source), ("target", self.target)):
            if any(not isinstance(node, int) or isinstance(node, bool) or node < 0 or node >= self.nodes for node in endpoints):
                raise ValueError(f"{name} node IDs must be integers in [0, {self.nodes})")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.timestamp):
            raise ValueError("timestamps must be non-negative integers")
        if any(left > right for left, right in zip(self.timestamp, self.timestamp[1:])):
            raise ValueError("timestamps must be nondecreasing")

    @property
    def edges(self) -> int:
        return len(self.source)

    def prefix(self, cutoff: int) -> TemporalEdges:
        """Return the events with timestamps strictly before ``cutoff``."""
        if not isinstance(cutoff, int) or isinstance(cutoff, bool):
            raise ValueError("cutoff must be an integer timestamp")
        stop = bisect_left(self.timestamp, cutoff)
        return TemporalEdges(
            self.nodes,
            self.source[:stop],
            self.target[:stop],
            self.timestamp[:stop],
        )


@dataclass(frozen=True, eq=False)
class StaticGraphTemporalSignal(Sequence[tuple[Tensor, Tensor]]):
    """An ordered tensor signal over one immutable graph."""

    graph: Graph
    node_ids: tuple[str, ...]
    x: Tensor
    y: Tensor
    edge_weight: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        if self.x.ndim != 3:
            raise ValueError(f"x must have shape [T, N, F], got {self.x.shape}")
        if self.y.ndim != 3:
            raise ValueError(f"y must have shape [T, N, Y], got {self.y.shape}")
        if self.x.shape[:2] != self.y.shape[:2]:
            raise ValueError("x and y must have the same time and node axes")
        if self.x.shape[1] != self.graph.nodes:
            raise ValueError(f"x must have {self.graph.nodes} node rows, got {self.x.shape[1]}")
        if self.x.shape[2] == 0 or self.y.shape[2] == 0:
            raise ValueError("feature counts must be positive")
        if len(self.node_ids) != self.graph.nodes:
            raise ValueError(f"expected {self.graph.nodes} node IDs, got {len(self.node_ids)}")
        if any(not isinstance(node_id, str) or not node_id for node_id in self.node_ids):
            raise ValueError("node IDs must be non-empty strings")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node IDs must be unique")
        if not dtypes.is_float(self.x.dtype):
            raise ValueError(f"x must have a floating dtype, got {self.x.dtype}")
        if not dtypes.is_float(self.y.dtype):
            raise ValueError(f"y must have a floating dtype, got {self.y.dtype}")
        if self.x.dtype != self.y.dtype:
            raise ValueError("x and y must share one dtype")
        if self.x.device != self.y.device or not isinstance(self.x.device, str):
            raise ValueError("x and y must share one device")
        if self.edge_weight is not None:
            if self.edge_weight.ndim != 1 or self.edge_weight.shape[0] != self.graph.edges:
                raise ValueError(
                    f"edge_weight must have shape [{self.graph.edges}], got {self.edge_weight.shape}"
                )
            if not dtypes.is_float(self.edge_weight.dtype):
                raise ValueError(f"edge_weight must have a floating dtype, got {self.edge_weight.dtype}")
            if self.edge_weight.dtype != self.x.dtype or self.edge_weight.device != self.x.device:
                raise ValueError("x and edge_weight must share dtype and device")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @overload
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...

    @overload
    def __getitem__(self, index: slice) -> StaticGraphTemporalSignal: ...

    def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor] | StaticGraphTemporalSignal:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("temporal slices must be contiguous and forward")
            return StaticGraphTemporalSignal(
                self.graph,
                self.node_ids,
                self.x[start:stop],
                self.y[start:stop],
                self.edge_weight,
            )
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.x[index], self.y[index]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        return (self[index] for index in range(len(self)))

    def split(self, train_ratio: float) -> tuple[StaticGraphTemporalSignal, StaticGraphTemporalSignal]:
        """Split once along time, preserving order and topology."""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between zero and one")
        train_steps = int(train_ratio * len(self))
        if train_steps == 0 or train_steps == len(self):
            raise ValueError("train and test must both be non-empty")
        return self[:train_steps], self[train_steps:]

    def batches(self, *, batch_size: int, history: int) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield causal sequence-to-one windows as [B, L, N, F] and [B, N, Y]."""
        for name, value in (("batch_size", batch_size), ("history", history)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if history > len(self):
            raise ValueError(f"history must not exceed the {len(self)} time steps")

        windows = len(self) - history + 1
        for start in range(0, windows, batch_size):
            stop = min(start + batch_size, windows)
            yield (
                Tensor.stack(
                    *(self.x[start + lag:stop + lag] for lag in range(history)),
                    dim=1,
                ),
                self.y[start + history - 1:stop + history - 1],
            )
