# tinymesh

tinymesh is a tinygrad-native library for learning over sparse structures
through space and time.

A graph is the smallest mesh. Nodes carry tensor fields and edges say which
nodes may interact. Coordinates, higher-dimensional cells, and time can extend
that structure without replacing the sparse core.

tinymesh is experimental. It currently proves fixed-graph unit and
scalar-weighted sparse aggregation, metric edge geometry, target-normalized
sparse attention, and trainable mean-GraphSAGE, unweighted GCN, single- and
multi-head GAT, T-GCN, and Chebyshev GConvGRU callers on CPU and Metal. Its
top-level 0.x surface is `Graph` and `StaticGraphTemporalSignal`; the dataset
module also exposes revision-bound loaders. The contract is intentionally
narrow and not stable.

## Try it

From a repository checkout:

```console
uv sync --locked
```

```python
from tinygrad import Tensor
from tinymesh import Graph

graph = Graph(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

print(graph.sum(state).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Node `2` receives values from nodes `0` and `1`. Node `3` receives the value
from node `1`. The other nodes have no incoming edges, so their aggregate is
zero.

The [quick start](quickstart.md) follows this value through topology lowering,
sparse execution, backward propagation, and five trainable layer families.

## The stack

```text
edge facts             source -> target, optional COO-aligned scalar value
    |
    v
topology lowering      COO -> CSR(A) + CSR(A.T) + edge maps
    |
    v
sparse edge operation  endpoint projection, target softmax
    |
    v
metric composition     position -> displacement -> distance -> edge weight
    |
    v
sparse aggregation     unit or scalar-weighted sum
    |
    v
directed diffusion     source-normalized forward + reverse propagation
    |
    v
temporal alignment     one Graph + x[T,N,F] + y[T,N,Y]
    |
    v
causal windows         values[B,L,N,F] + target[B,N,Y]
    |
    v
model composition      GraphSAGE, GCN, GAT, LSTM, T-GCN, GConvGRU, DiffusionGRU
```

[tinygrad](https://github.com/tinygrad/tinygrad) owns tensors, autograd,
compilation, and device execution. tinymesh owns sparse topology, mesh
semantics, and the model compositions that need them. It is not a PyTorch
Geometric compatibility layer.

The current core stores `O(N + E)` topology. Sum performs `O((N + E)H)` work for
`N` nodes, `E` edges, and feature width `H` without an `[N, N]` or `[E, H]`
carrier. Endpoint projection intentionally returns its declared `[E, H]` edge
field; no operation constructs an `N * E` axis.

## What works

- deterministic directed edge-list to CSR lowering;
- destination-owned sparse sum with no atomic writes;
- the same operation over transpose CSR for first-order backward;
- COO-ordered scalar edge weights with one gradient owner per edge;
- COO-ordered source and target projection with sparse backward;
- stable target-grouped softmax over scalar edge scores;
- metric displacement, distance, and radial weighting composed without a new
  public type or geo dependency;
- source-normalized scalar affinity propagated sparsely in both graph
  directions;
- fixed-topology device-buffer reuse;
- one mean-GraphSAGE composition whose neighbor parameter learns through the
  sparse boundary;
- one GCN composition with source and destination degree normalization;
- single- and multi-head GAT compositions whose attention parameters learn
  through endpoint projection, softmax, and weighted sum;
- one T-GCN composition whose hidden state and parameter gradient cross space
  and time;
- one GConvGRU composition that applies sparse Chebyshev filters to input and
  hidden state;
- one fixed-graph temporal signal with causal slicing and aligned node, feature,
  target, and edge axes;
- one shared-graph batch path that folds leading tensor lanes into sparse
  feature width and yields causal sequence-to-one windows;
- one pinned PyG Temporal chickenpox loader with no PyTorch or NumPy runtime;
- one pinned Montevideo loader that aligns directed topology, raw hourly
  signals, projected position, road distance, and one fixed coordinate frame;
- one Montevideo protocol with target-time splits, train-only per-node
  normalization, and raw-unit zero, persistence, and train-mean baselines;
- one matched three-seed Montevideo forecast where coordinate and road
  diffusion do not beat unit diffusion and all learned models lose to
  persistence;
- one three-seed Chickenpox forecast comparing matched LSTM, T-GCN, and
  GConvGRU models without finding a stable graph advantage;
- CPU and Metal tests.

`Graph` owns semantic COO identity under `src/tinymesh/`; its private CSR
backend owns lowering and device caches. `StaticGraphTemporalSignal` owns the
small fixed-topology data boundary. Model callers remain experiments.
The backend uses an alpha tinygrad surface and disables default kernel
optimization for its data-dependent loop. Vectorized attention heads, external
vector edge features, batching different graphs, changing topology,
higher-order gradients, timestamps, masks, general coordinate-reference
machinery, geodesy, and cells are not implemented.

## Learn how it works

- [Sparse graph topology](concepts/topology.md) explains COO, CSR, transpose,
  lowering, and the push-pull tradeoff.
- [Message passing](concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Time](concepts/time.md) explains fixed-topology snapshots, causal recurrence,
  temporal resolution, and why missingness is not zero.
- [Spatial structure](research/spatial-structure.md) separates physical
  connectivity, coordinate frames, node positions, and derived edge geometry.
- [Spatial geometry experiment](research/spatial-geometry.md) records the
  zero-geo-dependency metric composition, symmetries, gradients, and sparse
  UOps.
- [Directed diffusion experiment](research/directed-diffusion.md) records
  source normalization, reverse-edge identity, gradients, and sparse work.
- [Sparse aggregation feasibility](research/sparse-aggregation.md) records the
  revision-bound implementation and scaling evidence.
- [Mean GraphSAGE experiment](research/mean-sage.md) records the exact learning
  witness and its limits.
- [GCN experiment](research/gcn.md) records the normalized second caller and
  the shared boundary it exposes.
- [Weighted aggregation experiment](research/weighted-aggregation.md) records
  scalar edge identity and both first-order gradients.
- [Sparse attention experiment](research/attention.md) records endpoint
  projection, target softmax, and independently trainable heads.
- [T-GCN experiment](research/tgcn.md) records fixed-graph recurrence and the
  exact spatial-temporal learning witness.
- [GConvGRU experiment](research/gconv-gru.md) records the Chebyshev operator,
  fused recurrent cell, and controlled T-GCN comparison.
- [Chickenpox temporal data](research/chickenpox-data.md) records the first
  external dataset, its lowering, framework parity, and window contract.
- [Montevideo spatial-temporal data](research/montevideo-data.md) records the
  bounded real source and its topology, geometry, edge, and time contracts.
- [Montevideo forecast](research/montevideo-forecast.md) records target-time
  splits, a fused directed recurrent cell, and the negative geometry result.
- [Chickenpox forecast](research/chickenpox-forecast.md) records the first real
  training comparison, including the node-local controls and negative result.

Concept pages describe ideas that should survive an implementation change.
Research records bind claims to exact revisions and measurements. Source and
tests own current behavior.
