# tinymesh

tinymesh is a tinygrad-native library for learning over sparse structures
through space and time.

A graph is the smallest mesh. Nodes carry tensor fields and edges say which
nodes may interact. Coordinates, higher-dimensional cells, and time can extend
that structure without replacing the sparse core.

tinymesh is experimental. It currently proves fixed-graph unit and
scalar-weighted sparse aggregation, target-normalized sparse attention, and
trainable mean-GraphSAGE, unweighted GCN, and single- and multi-head GAT callers
on CPU and Metal. Its one public 0.x type is `Graph`; the contract is
intentionally narrow and not stable.

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
sparse execution, backward propagation, and three trainable layers.

## The stack

```text
edge facts             source -> target, optional COO-aligned scalar value
    |
    v
topology lowering      COO -> CSR(A) + CSR(A.T) + edge maps
    |
    v
sparse operation       sum, endpoint projection, target softmax
    |
    v
model composition      mean GraphSAGE, unweighted GCN, GAT heads
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
- fixed-topology device-buffer reuse;
- one mean-GraphSAGE composition whose neighbor parameter learns through the
  sparse boundary;
- one GCN composition with source and destination degree normalization;
- single- and multi-head GAT compositions whose attention parameters learn
  through endpoint projection, softmax, and weighted sum;
- CPU and Metal tests.

`Graph` owns semantic COO identity under `src/tinymesh/`; its private CSR
backend owns lowering and device caches. Model callers remain experiments.
The backend uses an alpha tinygrad surface and disables default kernel
optimization for its data-dependent loop. Vectorized head execution, external
vector edge features, batching, changing topology, higher-order gradients,
geometry, cells, and time are not implemented.

## Learn how it works

- [Sparse graph topology](concepts/topology.md) explains COO, CSR, transpose,
  lowering, and the push-pull tradeoff.
- [Message passing](concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
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

Concept pages describe ideas that should survive an implementation change.
Research records bind claims to exact revisions and measurements. Source and
tests own current behavior.
