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

[![Tests](https://github.com/spatioterra-ai/tinymesh/actions/workflows/tests.yml/badge.svg)](https://github.com/spatioterra-ai/tinymesh/actions/workflows/tests.yml)

## What works

The repository currently proves one narrow sparse core:

```text
COO connectivity + scalar edge facts
                  |
                  v
       CSR(A) + CSR(A.T) + edge maps
                  |
          +-------+--------+--------------------+
          |                |                    |
          v                v                    v
      unit sum        weighted sum       endpoint values
      A @ X           A_w @ X            + target softmax
      A.T @ dY        A_w.T @ dY + dw    + weighted sum
```

- An unordered directed edge list lowers deterministically into destination CSR
  for forward propagation and transpose CSR for backward propagation.
- Edge-aligned scalar values and their gradients stay in original COO order;
  private maps connect them to both CSR traversals.
- Numeric node positions compose through endpoint projection, displacement,
  distance, radial weighting, and sparse sum without a new public type or geo
  dependency.
- Sparse sum stores `O(N + E)` topology and each direction performs
  `O((N + E)H)` work for `N` nodes, `E` edges, and feature width `H`.
- Leading batch and time axes fold into feature width, so `Graph.sum` applies
  one shared sparse graph to `[..., N, H]` without copying topology.
- One tinygrad custom kernel implements both `A @ X` and `A.T @ dY`; weighted
  execution reuses it and computes `dw` with one owner per edge. Neither sum
  path constructs `[N, N]` or `[E, H]` intermediates.
- A mean-GraphSAGE experiment sends gradients through the sparse boundary into
  a neighbor parameter on CPU and Metal.
- An unweighted GCN experiment composes source and destination degree scaling
  around the same sparse sum.
- Edge endpoint projection and target-grouped stable softmax compose trainable
  single- and multi-head GAT experiments without an `N * E` axis.
- A T-GCN experiment reuses one graph across ordered node snapshots and sends a
  parameter gradient through space and time.
- A GConvGRU experiment adds sparse Chebyshev filtering over both input and
  hidden state and reports its extra parameter and sparse-call cost against
  T-GCN.
- A fixed-graph temporal signal and pinned PyG Temporal chickenpox loader keep
  graph, node, time, feature, target, and edge axes aligned on tinygrad tensors.
- A pinned Montevideo loader adds raw hourly values, projected node positions,
  observed road distance, and a fixed coordinate frame without a geo runtime.
- A directed-diffusion experiment source-normalizes positive scalar affinity in
  both graph directions with two sparse sums per application.
- Causal window batches feed node-local LSTM, T-GCN, and GConvGRU forecasts.
  On the first three-seed Chickenpox run, LSTM and GConvGRU are tied; the
  graph-recurrent cell has no stable quality advantage yet.
- Each `Graph` owns and reuses private realized connectivity; incoming degree
  stays a lazy difference of its CSR row pointers.

The public surface is experimental 0.x code, not a stability promise: the
private CSR backend uses alpha `Tensor.custom_kernel` and tinygrad's default
kernel optimization does not yet accept its data-dependent loop.

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

Run the six trainable witnesses:

```console
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
DEV=CPU uv run python -m experiments.gcn
DEV=METAL uv run python -m experiments.gcn
DEV=CPU uv run python -m experiments.gat
DEV=METAL uv run python -m experiments.gat
DEV=CPU uv run python -m experiments.multi_head_gat
DEV=METAL uv run python -m experiments.multi_head_gat
DEV=CPU uv run python -m experiments.tgcn
DEV=METAL uv run python -m experiments.tgcn
DEV=CPU uv run python -m experiments.gconv_gru
DEV=METAL uv run python -m experiments.gconv_gru
```

Inspect the pinned external temporal dataset:

```console
DEV=CPU uv run python -m experiments.chickenpox_data
DEV=METAL uv run python -m experiments.chickenpox_data
```

Inspect the pinned spatial-temporal dataset:

```console
DEV=CPU uv run python -m experiments.montevideo_data
DEV=METAL uv run python -m experiments.montevideo_data
```

Train the controlled Chickenpox forecast:

```console
DEV=CPU uv run python -m experiments.chickenpox_forecast
```

Inspect weighted forward and both first-order gradients:

```console
DEV=CPU uv run python -m experiments.weighted_aggregation
DEV=METAL uv run python -m experiments.weighted_aggregation
```

Inspect metric geometry and gradients:

```console
DEV=CPU uv run python -m experiments.spatial_geometry
DEV=METAL uv run python -m experiments.spatial_geometry
```

Inspect source-normalized propagation in both graph directions:

```console
DEV=CPU uv run python -m experiments.directed_diffusion
DEV=METAL uv run python -m experiments.directed_diffusion
```

## Learn

Start with the [documentation](https://spatioterra-ai.github.io/tinymesh/) or
run the [quick start](docs/quickstart.md):

- [Sparse graph topology](docs/concepts/topology.md) explains COO, CSR,
  transpose, lowering, and the push-pull tradeoff.
- [Message passing](docs/concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Time](docs/concepts/time.md) separates ordered node fields from fixed,
  weighted, and changing topology.
- [Spatial structure](docs/research/spatial-structure.md) separates physical
  connectivity, coordinate frames, node positions, and derived edge geometry.
- [Spatial geometry experiment](docs/research/spatial-geometry.md) proves
  direct metric composition, symmetry contracts, gradients, and sparse UOps.
- [Directed diffusion experiment](docs/research/directed-diffusion.md) proves
  sparse source normalization in both directions of a fixed graph.
- [Sparse aggregation feasibility](docs/research/sparse-aggregation.md) retains
  the revision-bound scaling and kernel evidence.
- [Mean GraphSAGE experiment](docs/research/mean-sage.md) retains the exact
  learning witness and its limits.
- [GCN experiment](docs/research/gcn.md) tests the same sparse sum under
  symmetric degree normalization.
- [Weighted aggregation experiment](docs/research/weighted-aggregation.md)
  follows scalar edge identity through lowering, forward, and both gradients.
- [Sparse attention experiment](docs/research/attention.md) follows node
  coefficients into COO edge order, target softmax, weighted aggregation, and
  independently trainable attention heads.
- [T-GCN experiment](docs/research/tgcn.md) follows hidden state and a parameter
  gradient through one spatial edge and one temporal transition.
- [GConvGRU experiment](docs/research/gconv-gru.md) compares node-local and
  graph-convolutional recurrence under one controlled temporal witness.
- [Chickenpox temporal data](docs/research/chickenpox-data.md) lowers one
  canonical PyG Temporal dataset into the public fixed-graph signal.
- [Montevideo spatial-temporal data](docs/research/montevideo-data.md) aligns
  real directed topology, position, distance, and hourly node fields.
- [Chickenpox forecast](docs/research/chickenpox-forecast.md) follows causal
  batches through three recurrent models and retains the inconclusive graph
  comparison.

## Direction

Unit and scalar-weighted sums now share deterministic topology plus
destination-CSR execution. `Graph` exposes ordered edge identity, incoming sum,
endpoint projection, target softmax, and in-degree; its private backend owns
lowering and rebuildable device caches. A fixed-graph temporal signal owns
aligned node IDs, features, targets, edge weights, and causal window batches.
The alpha kernel and optimizer boundary still block stability. Vectorized
attention heads, external vector edge features, batching different graphs,
changing topology, timestamps, and masks remain unimplemented.

Numeric coordinates now compose as ordinary node tensors. One dataset-specific
record carries a fixed coordinate frame and unit; general coordinate-reference
machinery, geodesy, higher-dimensional cells, and richer temporal fields remain
the wider mesh direction. They enter only when the sparse graph core extends
naturally; tinymesh is not a GIS, trainer framework, application, or model zoo.

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
PyTorch Geometric, PyTorch Geometric Temporal, TorchGeo, and TerraTorch:

```console
git submodule update --init
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
