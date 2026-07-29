"""Neural network layers composed from tinygrad and sparse mesh operations."""

from math import sqrt

from tinygrad import Tensor, nn

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
