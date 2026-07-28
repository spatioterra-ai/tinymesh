# Weighted aggregation experiment

This record asks whether one scalar per COO edge can survive topology lowering,
sparse forward, and both first-order gradients without creating edge-feature
state.

## Decision

At tinygrad revision
[`0bb36c9`](https://github.com/tinygrad/tinygrad/tree/0bb36c99899ba4742dbe1d5784397373998d81c3),
the fixed-topology CSR experiment computes:

```text
Y_v  = sum(w_e X_u)          for every edge e: u -> v
dX_u = sum(w_e dY_v)
dw_e = dot(X_u, dY_v)
```

on CPU and Metal. Forward and `dX` reuse destination-owned CSR traversal. `dw`
gives one output to each edge and reduces only feature width. None of the three
paths needs atomics or a materialized `[E, H]` tensor.

The result earns scalar edge multiplication in the experimental `Graph.sum`
contract. It does not earn API stability: the implementation still relies on
alpha `Tensor.custom_kernel`, and the data-dependent CSR loop still disables
tinygrad's default kernel optimization.

## Edge order is explicit

Weights arrive in the same order as the `source` and `target` arrays passed to
`Graph`. Connectivity sorting must not detach them:

```text
raw COO weight
    |                         |
    | edge_order             | transpose_order
    v                         v
forward CSR lookup       transpose CSR lookup
    |                         |
    v                         v
Y = A_w @ X              dX = A_w.T @ dY

raw source + target
    |
    v
dw_e = dot(X_source[e], dY_target[e])
```

Both maps point from a CSR position to the original COO ordinal. Kernels use
them internally; the input weight tensor and its gradient never leave COO
order. Duplicate source-target pairs keep separate positions.

The two CSR forms still contain `2E + 2(N + 1)` integers. Edge identity retains
original source and target plus two order maps. Weighted execution realizes
those `4E` device integers and receives `E` scalar weights. Storage remains
`O(N + E)`.

## Exact witness

The runnable fixture uses:

```text
COO edge:     0 -> 2    1 -> 2    0 -> 1
weight:          2         -1          3
node value:   X_0 = 2    X_1 = 4    X_2 = 8
output grad:  g_0 = 0    g_1 = 5    g_2 = 7
```

Connectivity lowering produces `edge_order = (2, 0, 1)`, but the weights stay
in the listed COO order. Both devices return:

```text
Y               = [0, 6, 0]
dX              = [29, -7, 0]
dw in COO order = [14, 28, 10]
```

The multidimensional test suite compares forward, `dX`, and `dw` with an
independent host edge loop. It also covers paired COO permutations, duplicate
edges with different weights, isolated and empty rows, the one-node case,
shape, dtype, and device rejection.

Structural checks inspect the generated UOps. Forward and `dX` each launch
`N * H` owners with one dynamic CSR row loop. `dw` launches `E` owners with one
static reduction of width `H`. The largest tensor-shaped intermediate is
bounded by node state, edge state, or row pointers; no `[E, H]` carrier appears.
This proves `O((N + E)H)` forward and `dX` work and `O(EH)` edge-gradient work.
It is not a speed claim.

## Reference contract

PyTorch Geometric 2.8.0 `GraphConv` accepts `edge_weight` beside `edge_index`
and defines its message as `edge_weight * x_j`; compatible inputs can fuse that
operation into sparse matrix multiplication
([source](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/conv/graph_conv.py#L77-L110)).
PyG receives weights in the current edge-index order on each call. tinymesh
now exposes the same ordering rule while a fixed topology owns private CSR
lookup maps.

PyTorch Geometric Temporal carries one edge-weight array with a static
edge-index array
([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/signal/static_graph_temporal_signal.py#L31-L120))
or aligned sequences of both for a dynamic graph
([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/signal/dynamic_graph_temporal_signal.py#L30-L125)).
That is the next data-contract question, not evidence for a temporal kernel.
tinymesh currently proves one fixed weighted topology only.

## Limits

The experiment covers scalar weights, one device, fixed topology, and
first-order gradients. Vector edge features, learned attention scores, segment
normalization, batching, topology changes, higher-order gradients, temporal
snapshots, and model quality remain unproven. The edge-weight kernel reduces
feature width serially per edge; degree skew still affects the CSR directions.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_weighted_aggregation
DEV=METAL uv run python -m unittest tests.test_weighted_aggregation
DEV=CPU uv run python -m experiments.weighted_aggregation
DEV=METAL uv run python -m experiments.weighted_aggregation
```
