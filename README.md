<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/assets/logo_mesh_light.svg">
  <img alt="tinymesh" src="/docs/assets/logo_mesh_dark.svg" width="50%" height="50%">
</picture>

</div>

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
- `SAGEConv`, `GCNConv`, and `GATConv` compose direct tinygrad parameters over
  the same sparse graph operations.
- `TGCN`, `A3TGCN`, `ChebConv`, and `GConvGRU` reuse one graph across ordered
  node snapshots and send parameter gradients through space and time.
- A fixed-graph temporal signal and pinned PyG Temporal chickenpox loader keep
  graph, node, time, feature, target, and edge axes aligned on tinygrad tensors.
- A pinned Montevideo loader adds raw hourly values, projected node positions,
  observed road distance, and a fixed coordinate frame without a geo runtime.
- A checksummed METR-LA loader aligns 34,272 five-minute rows, explicit
  zero-sentinel missingness, 207 sensor IDs, and 1,722 directed affinity edges
  without NumPy, pandas, pickle, or a dense adjacency tensor.
- A leakage-safe METR-LA protocol reproduces PyG Temporal's 12-to-12 A3T-GCN
  task shape, then splits by target time, fits observed training rows only,
  masks missing targets, and establishes persistence at test MAE 4.232 and
  RMSE 8.145 as the graph-model floor.
- A persistence-anchored residual A3T-GCN improves validation MAE by 0.5% and
  RMSE by 1.1%, but self-only topology beats the real and permuted graphs in
  all three seeds; the current gain is temporal, not spatial.
- `DirectedDiffusion` and `DiffusionGRU` source-normalize caller-validated
  positive scalar affinity in both graph directions and keep recurrent
  propagation sparse.
- A Montevideo forecast protocol preserves target time, fits per-node
  normalization on training rows, and reports raw-unit zero, persistence, and
  train-mean baselines.
- A validation-selected, train-only hour-of-week baseline improves Montevideo
  validation RMSE by 27.1% and test RMSE by 32.8% over persistence without
  using topology.
- A delayed-residual study finds no metric-consistent advantage for real
  directed edges over the seasonal floor, reversed edges, or permuted node
  fields; it adds no new API.
- A fused bidirectional diffusion GRU trains with unit, coordinate-distance,
  and road-distance affinity. In the frozen three-seed Montevideo comparison,
  neither geometry variant beats unit diffusion and every learned model loses
  to persistence.
- Causal window batches feed node-local LSTM, T-GCN, and GConvGRU forecasts.
  On the first three-seed Chickenpox run, LSTM and GConvGRU are tied; the
  graph-recurrent cell has no stable quality advantage yet.
- A controlled 24-node transport witness isolates known spatial propagation.
  True-topology DiffusionGRU reduces validation one-step RMSE by 80.2% and
  four-step rollout RMSE by 79.3% against a degree-matched false graph, winning
  both comparisons in all three seeds.
- The same frozen DiffusionGRU transfers to unseen 32- and 48-node graphs. It
  beats permuted topology, self-only topology, a node-local LSTM, and
  persistence in all three seeds across dense and localized fields; its
  16-step rollout improvement over the strongest alternative is 20.1% to
  30.4%.
- Each `Graph` owns and reuses private realized connectivity; incoming degree
  stays a lazy difference of its CSR row pointers.

`Graph` and `StaticGraphTemporalSignal` are top-level types. `tinymesh.nn` owns
the proven reusable components and `tinymesh.datasets` owns pinned loaders.
This is experimental 0.x code, not a stability promise: the private CSR backend
uses alpha `Tensor.custom_kernel` and tinygrad's default kernel optimization
does not yet accept its data-dependent loop.

## Try it

Install the exact locked tinygrad revision with
[uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
```

Run one sparse aggregation:

```python
from tinygrad import Device, Tensor
from tinymesh import Graph

graph = Graph(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device=Device.DEFAULT).realize()

print(graph.sum(state).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Layers are direct tinygrad-style objects:

```python
from tinymesh.nn import SAGEConv

layer = SAGEConv(in_features=1, out_features=2)
print(layer(state, graph).shape)
# (4, 2)
```

List or record the revision-bound evidence:

```console
uv run --locked python -m experiments.run --list
uv run --locked python -m experiments.run mean_sage DEV=CPU
uv run --locked python -m experiments.run tgcn DEV=METAL
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU EPOCHS=10 SEED=0
uv run --locked python -m experiments.run metr_la_forecast DEV=CPU
```

Successful runs write ignored local envelopes containing the tinymesh revision,
all five reference pins, explicit settings, execution bounds, and the JSON
observation. See [Experiments](docs/experiments.md).

## Learn

Start with the [documentation](https://spatioterra-ai.github.io/tinymesh/) or
run the [quick start](docs/quickstart.md):

- [Sparse graph topology](docs/concepts/topology.md) explains COO, CSR,
  transpose, lowering, and the push-pull tradeoff.
- [Message passing](docs/concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Time](docs/concepts/time.md) separates ordered node fields from fixed,
  weighted, and changing topology.
- [Experiments](docs/experiments.md) explains the catalog, local run ledger,
  and component graduation.
- [Reference projects](docs/reference-projects.md) records how tinygrad, PyG,
  PyG Temporal, TorchGeo, and TerraTorch influence the design.
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
- [METR-LA sensor data](docs/research/metr-la-data.md) aligns raw traffic
  speed, source timestamps, missingness, and directed road-distance affinity.
- [METR-LA forecast](docs/research/metr-la-forecast.md) reproduces the A3T-GCN
  task shape, fixes leakage and missingness policy, and establishes the
  temporal floor plus current execution boundary.
- [Montevideo forecast](docs/research/montevideo-forecast.md) fixes target-time
  splits and retains the negative matched geometry comparison.
- [Montevideo seasonal floor](docs/research/montevideo-seasonal.md) selects the
  temporal control a later graph experiment must beat.
- [Montevideo delayed edges](docs/research/montevideo-delayed-edges.md) tests
  causal graph residuals against reversed-edge and permuted-field controls.
- [Chickenpox forecast](docs/research/chickenpox-forecast.md) follows causal
  batches through three recurrent models and retains the inconclusive graph
  comparison.
- [Controlled transport transfer](docs/research/transport-transfer.md) tests
  one frozen local operator on unseen graph sizes, localized fields, structural
  controls, and longer rollouts.

## Direction

Unit and scalar-weighted sums now share deterministic topology plus
destination-CSR execution. `Graph` exposes ordered edge identity, incoming sum,
incoming mean, endpoint projection, target softmax, and in-degree; its private
backend owns lowering and rebuildable device caches. A fixed-graph temporal
signal owns aligned node IDs, features, targets, edge weights, and causal
window batches. A separate dataset-specific METR-LA record retains raw
timestamps and a derived missingness mask before task policy exists; its
forecast protocol remains research-owned.
`tinymesh.nn` owns equations and parameters; experiments own data policy,
unrolling, training, controls, and claims.
The alpha kernel and optimizer boundary still block stability. Vectorized
attention heads, external vector edge features, batching different graphs,
changing topology, and generic timestamp and missingness containers remain
unimplemented.

Numeric coordinates now compose as ordinary node tensors. One dataset-specific
record carries a fixed coordinate frame and unit; general coordinate-reference
machinery, geodesy, higher-dimensional cells, and richer temporal fields remain
the wider mesh direction. They enter only when the sparse graph core extends
naturally; tinymesh is not a GIS, trainer framework, application, or model zoo.

The first matched directed real-data forecast found no stable value from fixed
scalar distance. A later temporal study found a much stronger hour-of-week
floor, and the first delayed-edge residual test did not beat that floor or
structural controls. Controlled transport now distinguishes the existing
sparse diffusion path and shows size transfer under one known law. Further
synthetic tuning has diminishing value. METR-LA now supplies the real sensor
boundary, a trustworthy 12-to-12 protocol, and strong temporal controls.
Full-size A3T-GCN training now fits the local 600-second bound at batch 512.
The matched METR-LA result finds a small residual temporal gain while rejecting
a spatial advantage from unit-weight GCN neighbor mixing.

## Development

```console
uv sync --locked
uv run --locked python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Build or preview the documentation with the locked docs environment:

```console
uv run --locked --only-group docs zensical build --clean --strict
uv run --locked --only-group docs zensical serve
```

The pinned submodules are optional, reference-only source:

```console
git submodule update --init
```

Their exact roles and exclusions live in
[Reference projects](docs/reference-projects.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
