# tinymesh

**Sparse structure through space and time, in tinygrad.**

tinymesh is an experimental library for learning over graphs and meshes with
[tinygrad](https://github.com/tinygrad/tinygrad). A graph is the smallest mesh:
sparse topology connects tensor fields. Geometry and time can extend that core
without changing it. tinymesh is tinygrad-native, not a compatibility layer
over another machine-learning framework.

[Documentation](https://spatioterra-ai.github.io/tinymesh/) |
[Quick start](https://spatioterra-ai.github.io/tinymesh/quickstart/) |
[Contributing](CONTRIBUTING.md)

## What works

The repository currently proves one narrow sparse core:

```text
COO connectivity + scalar edge facts
                  |
                  v
       CSR(A) + CSR(A.T) + edge maps
                  |
          +-------+--------+
          |                |
          v                v
      unit sum        weighted sum
      A @ X           A_w @ X
      A.T @ dY        A_w.T @ dY + dw
```

- An unordered directed edge list lowers deterministically into destination CSR
  for forward propagation and transpose CSR for backward propagation.
- Edge-aligned scalar values and their gradients stay in original COO order;
  private maps connect them to both CSR traversals.
- Sparse sum stores `O(N + E)` topology and each direction performs
  `O((N + E)H)` work for `N` nodes, `E` edges, and feature width `H`.
- One tinygrad custom kernel implements both `A @ X` and `A.T @ dY`; weighted
  execution reuses it and computes `dw` with one owner per edge. No path
  constructs `[N, N]` or `[E, H]` intermediates.
- A mean-GraphSAGE experiment sends gradients through the sparse boundary into
  a neighbor parameter on CPU and Metal.
- An unweighted GCN experiment composes source and destination degree scaling
  around the same sparse sum.
- Each `Graph` owns and reuses its realized connectivity and degree buffers.

`Graph` is Tinymesh's first public API. It is experimental 0.x code, not a
stability promise: the private CSR backend uses alpha `Tensor.custom_kernel`
and tinygrad's default kernel optimization does not yet accept its
data-dependent loop.

## Run the proof

Install the exact locked tinygrad revision with
[uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
```

Then run one sparse aggregation from the repository checkout:

```python
from tinygrad import Tensor
from tinymesh import Graph

graph = Graph(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

print(graph.sum(state).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Both trainable witnesses start with loss `1`, take one SGD step, and reach loss
`0` through graph propagation:

```console
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
DEV=CPU uv run python -m experiments.gcn
DEV=METAL uv run python -m experiments.gcn
```

Inspect weighted forward and both first-order gradients:

```console
DEV=CPU uv run python -m experiments.weighted_aggregation
DEV=METAL uv run python -m experiments.weighted_aggregation
```

## Learn

Start with the [documentation](https://spatioterra-ai.github.io/tinymesh/) or
run the [quick start](docs/quickstart.md):

- [Sparse graph topology](docs/concepts/topology.md) explains COO, CSR,
  transpose, lowering, and the push-pull tradeoff.
- [Message passing](docs/concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Sparse aggregation feasibility](docs/research/sparse-aggregation.md) retains
  the revision-bound scaling and kernel evidence.
- [Mean GraphSAGE experiment](docs/research/mean-sage.md) retains the exact
  learning witness and its limits.
- [GCN experiment](docs/research/gcn.md) tests the same sparse sum under
  symmetric degree normalization.
- [Weighted aggregation experiment](docs/research/weighted-aggregation.md)
  follows scalar edge identity through lowering, forward, and both gradients.

## Direction

Unit and scalar-weighted sums now share deterministic topology plus
destination-CSR execution. `Graph` exposes only ordered edge identity, the
proven incoming sum, and in-degree; its private backend owns lowering and
rebuildable device caches. The alpha kernel and optimizer boundary still block
stability. Vector edge features, attention, batching, changing topology, and
temporal recurrence remain unimplemented.

Coordinates, coordinate-reference metadata, higher-dimensional cells, and
time-varying fields remain the wider mesh direction. They enter only when the
sparse graph core extends naturally; tinymesh is not a GIS, trainer framework,
application, or model zoo.

## Development

```console
uv sync --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Build or preview the documentation with the locked docs environment:

```console
uv run --locked --only-group docs zensical build --clean --strict
uv run --locked --only-group docs zensical serve
```

The pinned submodules are optional, reference-only source for studying tinygrad,
PyTorch Geometric, and PyTorch Geometric Temporal:

```console
git submodule update --init
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
