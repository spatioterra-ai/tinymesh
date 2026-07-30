"""Neural network layers composed from tinygrad and sparse mesh operations."""

from collections import Counter
from math import sqrt

from tinygrad import Tensor, dtypes, nn

from tinymesh.graph import Graph


class SAGEConv:
    """Mean GraphSAGE over one homogeneous graph."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.neighbor = nn.Linear(in_features, out_features, bias=bias)
        self.root = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        return self.neighbor(graph.mean(values)) + self.root(values)


class GCNConv:
    """Unweighted GCN over caller-supplied edges and self-loops."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        if not isinstance(values.device, str):
            raise ValueError("GCNConv requires one device")
        degree = graph.in_degree(device=values.device)
        scale = (degree != 0).where(
            degree.maximum(1).cast(values.dtype).rsqrt(),
            0,
        ).reshape((1,) * (values.ndim - 2) + (graph.nodes, 1))
        return self.linear(graph.sum(values * scale) * scale)


class GATConv:
    """Graph attention with independently normalized concatenated heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        if heads <= 0:
            raise ValueError("heads must be positive")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        bound = 1 / sqrt(out_features)
        self.source_attention = Tensor.uniform(heads, out_features, low=-bound, high=bound)
        self.target_attention = Tensor.uniform(heads, out_features, low=-bound, high=bound)
        self.bias: Tensor | None = Tensor.zeros(heads * out_features) if bias else None
        self.heads, self.out_features = heads, out_features
        self.negative_slope = negative_slope

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        if values.ndim != 2 or values.shape[0] != graph.nodes:
            raise ValueError(f"values must have shape [{graph.nodes}, F], got {values.shape}")
        state = self.linear(values).reshape(graph.nodes, self.heads, self.out_features)
        source_score = (state * self.source_attention).sum(axis=2)
        target_score = (state * self.target_attention).sum(axis=2)
        edge_score = (
            graph.edge_values(source_score, endpoint="source")
            + graph.edge_values(target_score, endpoint="target")
        ).leaky_relu(self.negative_slope)
        heads = [
            graph.sum(state[:, head], graph.softmax(edge_score[:, head]))
            for head in range(self.heads)
        ]
        output = heads[0].cat(*heads[1:], dim=1)
        return output if self.bias is None else output + self.bias


class ChebConv:
    """Chebyshev graph convolution for symmetric, loop-free unit edges."""

    def __init__(self, in_features: int, out_features: int, order: int) -> None:
        if in_features <= 0 or out_features <= 0 or order <= 0:
            raise ValueError("feature counts and order must be positive")
        self.linear = nn.Linear(order * in_features, out_features)
        self.in_features, self.order = in_features, order

    def __call__(self, values: Tensor, graph: Graph) -> Tensor:
        _validate_chebyshev_graph(graph)
        return self._project(values, graph)

    def _project(self, values: Tensor, graph: Graph) -> Tensor:
        expected = (graph.nodes, self.in_features)
        if values.ndim < 2 or values.shape[-2:] != expected:
            raise ValueError(f"values must have shape [..., {graph.nodes}, {self.in_features}], got {values.shape}")
        if not isinstance(values.device, str):
            raise ValueError("ChebConv requires one device")

        degree = graph.in_degree(device=values.device)
        scale = (degree != 0).where(
            degree.maximum(1).cast(values.dtype).rsqrt(),
            0,
        ).reshape((1,) * (values.ndim - 2) + (graph.nodes, 1))
        states = [values]
        if self.order > 1:
            states.append(_chebyshev_shift(values, graph, scale))
        for _ in range(2, self.order):
            states.append(2 * _chebyshev_shift(states[-1], graph, scale) - states[-2])
        basis = states[0] if self.order == 1 else states[0].cat(*states[1:], dim=-1)
        return self.linear(basis)


class TGCN:
    """One temporal graph convolutional recurrent step."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        self.graph_projection = GCNConv(in_features, 3 * hidden_features, bias=False)
        self.update = nn.Linear(2 * hidden_features, hidden_features)
        self.reset = nn.Linear(2 * hidden_features, hidden_features)
        self.candidate = nn.Linear(2 * hidden_features, hidden_features)
        self.hidden_features = hidden_features

    def __call__(self, values: Tensor, graph: Graph, hidden: Tensor | None = None) -> Tensor:
        graph_state = self.graph_projection(values, graph)
        update_input = graph_state[..., :self.hidden_features]
        reset_input = graph_state[..., self.hidden_features:2 * self.hidden_features]
        candidate_input = graph_state[..., 2 * self.hidden_features:]
        hidden = _hidden(update_input, hidden, self.hidden_features)
        update = self.update(update_input.cat(hidden, dim=-1)).sigmoid()
        reset = self.reset(reset_input.cat(hidden, dim=-1)).sigmoid()
        candidate = self.candidate(candidate_input.cat(hidden * reset, dim=-1)).tanh()
        return update * hidden + (1 - update) * candidate


class A3TGCN:
    """Attention over T-GCN encodings of a fixed number of periods."""

    def __init__(self, in_features: int, hidden_features: int, periods: int) -> None:
        if periods <= 0:
            raise ValueError("periods must be positive")
        self.cell = TGCN(in_features, hidden_features)
        self.attention = Tensor.uniform(periods)
        self.in_features, self.hidden_features, self.periods = in_features, hidden_features, periods

    def __call__(
        self,
        values: Tensor,
        graph: Graph,
        hidden: Tensor | None = None,
    ) -> Tensor:
        expected = (self.periods, graph.nodes, self.in_features)
        if values.ndim < 3 or values.shape[-3:] != expected:
            raise ValueError(f"values must have shape [..., {self.periods}, {graph.nodes}, {self.in_features}], got {values.shape}")
        probability = self.attention.softmax(axis=0)
        states = [
            self.cell(values[..., period, :, :], graph, hidden) * probability[period]
            for period in range(self.periods)
        ]
        return sum(states[1:], start=states[0])


class GConvGRU:
    """One Chebyshev graph-convolutional recurrent step."""

    def __init__(self, in_features: int, hidden_features: int, order: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        self.gates = ChebConv(in_features + hidden_features, 2 * hidden_features, order)
        self.candidate = ChebConv(in_features + hidden_features, hidden_features, order)
        self.in_features, self.hidden_features = in_features, hidden_features

    def __call__(self, values: Tensor, graph: Graph, hidden: Tensor | None = None) -> Tensor:
        _validate_chebyshev_graph(graph)
        expected = (graph.nodes, self.in_features)
        if values.ndim < 2 or values.shape[-2:] != expected:
            raise ValueError(f"values must have shape [..., {graph.nodes}, {self.in_features}], got {values.shape}")
        hidden = _hidden(values, hidden, self.hidden_features)

        gates = self.gates._project(values.cat(hidden, dim=-1), graph)
        update = gates[..., :self.hidden_features].sigmoid()
        reset = gates[..., self.hidden_features:].sigmoid()
        candidate = self.candidate._project(values.cat(hidden * reset, dim=-1), graph).tanh()
        return update * hidden + (1 - update) * candidate


class DirectedDiffusion:
    """Bidirectional propagation for caller-validated positive affinity."""

    def __init__(self, graph: Graph, affinity: Tensor) -> None:
        if affinity.ndim != 1 or affinity.shape[0] != graph.edges:
            raise ValueError(f"affinity must have shape [{graph.edges}], got {affinity.shape}")
        if not dtypes.is_float(affinity.dtype):
            raise ValueError(f"affinity must have a floating dtype, got {affinity.dtype}")
        if not isinstance(affinity.device, str):
            raise ValueError("directed diffusion requires one device")

        self.graph = graph
        self.reverse = Graph(graph.nodes, graph.target, graph.source)
        one = Tensor.ones(graph.nodes, 1, dtype=affinity.dtype, device=affinity.device)
        outgoing = self.reverse.sum(one, edge_weight=affinity)
        incoming = graph.sum(one, edge_weight=affinity)
        self.forward_weight = affinity / graph.edge_values(outgoing, endpoint="source").flatten()
        self.reverse_weight = affinity / graph.edge_values(incoming, endpoint="target").flatten()

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.graph.sum(values, edge_weight=self.forward_weight),
            self.reverse.sum(values, edge_weight=self.reverse_weight),
        )


class DiffusionGRU:
    """One gated recurrent step over bidirectional directed diffusion."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("feature counts must be positive")
        width = 3 * (in_features + hidden_features)
        self.gates = nn.Linear(width, 2 * hidden_features)
        self.candidate = nn.Linear(width, hidden_features)
        self.in_features, self.hidden_features = in_features, hidden_features

    def __call__(
        self,
        values: Tensor,
        diffusion: DirectedDiffusion,
        hidden: Tensor | None = None,
    ) -> Tensor:
        expected = (diffusion.graph.nodes, self.in_features)
        if values.ndim < 2 or values.shape[-2:] != expected:
            raise ValueError(
                f"values must have shape [..., {diffusion.graph.nodes}, "
                f"{self.in_features}], got {values.shape}"
            )
        hidden = _hidden(values, hidden, self.hidden_features)
        gates = self.gates(self._basis(values.cat(hidden, dim=-1), diffusion))
        update = gates[..., :self.hidden_features].sigmoid()
        reset = gates[..., self.hidden_features:].sigmoid()
        candidate = self.candidate(
            self._basis(values.cat(hidden * reset, dim=-1), diffusion)
        ).tanh()
        return update * hidden + (1 - update) * candidate

    def _basis(self, values: Tensor, diffusion: DirectedDiffusion) -> Tensor:
        forward, reverse = diffusion(values)
        return values.cat(forward, reverse, dim=-1)


def _hidden(values: Tensor, hidden: Tensor | None, features: int) -> Tensor:
    if hidden is None:
        return Tensor.zeros(
            *values.shape[:-1],
            features,
            dtype=values.dtype,
            device=values.device,
        )
    expected = (*values.shape[:-1], features)
    if hidden.shape != expected:
        raise ValueError(f"hidden must have shape {expected}, got {hidden.shape}")
    if hidden.dtype != values.dtype or hidden.device != values.device:
        raise ValueError("hidden and values must share dtype and device")
    return hidden


def _chebyshev_shift(values: Tensor, graph: Graph, scale: Tensor) -> Tensor:
    return -graph.sum(values * scale) * scale


def _validate_chebyshev_graph(graph: Graph) -> None:
    if any(source == target for source, target in zip(graph.source, graph.target)):
        raise ValueError("Chebyshev graphs must not contain self-loops")
    edges = Counter(zip(graph.source, graph.target))
    reverse = Counter((target, source) for source, target in zip(graph.source, graph.target))
    if edges != reverse:
        raise ValueError("Chebyshev graphs must be symmetric")
