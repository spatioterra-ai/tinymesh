# Sparse attention experiment

This record asks whether graph-attention layers can be ordinary tinygrad
composition over a few sparse graph primitives.

## Decision

At tinygrad revision
[`c9e1154`](https://github.com/tinygrad/tinygrad/tree/c9e11544df5db55c13f06d01fba5300dd44332fb),
Tinymesh computes single- and multi-head graph-attention layers and their
first-order gradients on CPU and Metal without node-pair or node-edge Cartesian
work.

The result adds two experimental `Graph` operations:

```text
edge_values(X, endpoint)  node field [N, H] -> COO edge field [E, H]
softmax(score)             COO edge score [E] -> target-normalized score [E]
```

The proven layer now lives in `tinymesh.nn.GATConv`; this experiment retains
the one- and two-head learning witnesses. Automatic self-loops, dropout,
residuals, edge-feature scoring, and vectorized head execution are not
implemented.

## Composition

One head transforms node state, computes one source and target coefficient per
node, gathers those scalar coefficients into edge order, normalizes them by
destination, and reuses weighted sum:

```text
X [N, Fin]
    |
    v
XW [N, Fout]
    |
    +--> source coefficient [N, 1] --+
    |                                 |
    +--> target coefficient [N, 1] --+--> edge score [E]
                                           |
                                     target softmax
                                           |
                                      attention [E]
                                           |
                              Graph.sum(XW, attention)
                                           |
                                           v
                                      output [N, Fout]
```

Self-loops remain explicit graph edges. Tinymesh does not silently change
topology inside the model.

## Multiple heads are composition

For `K` heads of width `C`, the linear map returns `[N, K, C]` node state and
the learned attention vectors return `[N, K]` node scores. Endpoint projection
lifts all head scores at once:

```text
state [N, K, C]
    |
    +--> node scores [N, K] --> edge_values --> edge scores [E, K]
    |                                               |
    |                         +---------------------+-------------------+
    |                         |                     |                   |
    |                         v                     v                   v
    |                    softmax head 0        softmax head 1          ...
    |                         |                     |
    +--> state head 0 --> weighted sum         weighted sum <-- state head 1
                              |                     |
                              +------ concatenate --+
                                          |
                                          v
                                     [N, K * C]
```

Each head calls the existing scalar `Graph.softmax` and `Graph.sum`. Head count
is a small model-construction constant, so the Python loop is explicit. This
keeps the public graph contract unchanged and makes kernel count grow linearly
with `K`; a vectorized core path needs performance evidence before it exists.

## Endpoint projection

For an edge `e: u -> v`:

```text
edge_values(X, "source")[e] = X[u]
edge_values(X, "target")[e] = X[v]
```

Forward gives each edge-feature lane one writer and performs one indexed load.
Backward groups edge gradients by the selected endpoint and uses the existing
CSR row sum:

```text
dX[u] = sum(dEdge[e] for each edge whose selected endpoint is u)
```

The output intentionally has shape `[E, H]`; that is the requested edge field,
not a hidden dense carrier. Work and output storage are `O(EH)`. No path
introduces an `N * E` axis.

The checked-in single-head caller projects node state to scalar source and
target coefficients before endpoint projection. Its score path therefore
materializes `[E, 1]`. The multi-head caller materializes the requested `[E, K]`
score field, not `[E, K, C]`.

## Target softmax

For each edge `e: u -> v`:

```text
alpha[e] = exp(score[e] - max_score[v])
alpha[e] = alpha[e] / sum(alpha[k] for every edge k ending at v)
```

The maximum is detached. Differentiating the shifted quotient while treating
the maximum as constant gives the same softmax Jacobian, while avoiding a
max-gradient rule and matching the stable sparse formulation used by PyTorch
Geometric.

Tinymesh composes four sparse steps:

```text
segment max by target
    -> gather target maximum to edges
    -> segment sum exponentials by target
    -> gather target total to edges
```

Both segment reductions traverse destination CSR. Both gathers return values in
original COO order. The forward visits rows or edges a constant number of
times, so work and stored state remain `O(N + E)`. High-degree rows still
serialize inside the current pull kernel.

## Exact single-head witness

The runnable model uses:

```text
node values:  X0 = 1, X1 = -1, X2 = 0
edges:        0 -> 2, 1 -> 2
target:       Y2 = 1
```

It starts with linear weight `1`, source-attention weight `1`, and
target-attention weight `0`. One SGD step at learning rate `0.1` returns on
both backends, to float32 precision:

```text
initial loss               0.214323
source-attention gradient -0.395310
final loss                 0.126819
linear weight              1.089257
source-attention weight    1.039531
```

This proves that a shared attention parameter receives a gradient through edge
projection, target softmax, and weighted CSR aggregation. It does not establish
model quality.

## Exact two-head witness

The second fixture uses the same two incoming values and gives its heads source
attention parameters `1` and `-1`. They attend in opposite directions and
produce two concatenated output channels. One SGD step returns on both backends:

```text
initial loss                     0.214323
source-attention gradient      [-0.197655,  0.197655]
final loss                       0.168497
linear weights                 [ 1.044628,  1.044628]
source-attention parameters    [ 1.019765, -1.019765]
```

The host reference normalizes each head independently. A separate head
permutation test swaps parameters and observes only swapped output columns.
Together these show that heads neither share normalization nor mix before
concatenation.

## Reference contract

PyTorch Geometric 2.8 computes source and destination coefficients at nodes,
lifts them to edges, applies LeakyReLU and sparse softmax, then multiplies
source messages by the result
([GATConv source](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/nn/conv/gat_conv.py#L328-L409)).
It represents transformed state as `[N, K, C]`, then concatenates head outputs
([multi-head source](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/nn/conv/gat_conv.py#L284-L366)).
Its stable sparse softmax detaches the segment maximum, exponentiates shifted
scores, and divides by a segment sum
([softmax source](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/utils/_softmax.py#L61-L91)).

Tinymesh adopts that mathematical decomposition, not PyG's framework surface.
`Graph` owns fixed topology and COO identity; ordinary tinygrad operations own
linear maps, coefficient calculation, LeakyReLU, exponentiation, division, and
optimization.

The formulation follows the
[Graph Attention Networks paper](https://arxiv.org/abs/1710.10903).

## Evidence and limits

Tests compare endpoint forward and backward with host edge loops; compare
softmax and its gradient with the closed-form grouped result; compare one and
two heads with independent host references; cover COO and head permutations,
duplicates, empty graphs, one-node graphs, isolated rows, large-score stability,
validation, and vertex permutation; and inspect UOp shapes and loop bounds for
forbidden dense carriers.

The result covers fixed topology, scalar per-head scores, concatenated heads,
one device, and first-order gradients. It does not cover vectorized head
normalization, head averaging, learned external edge features, batching,
changing topology, higher-order gradients, temporal recurrence, or useful
predictive accuracy.
Endpoint projection materializes its declared `[E, H]` output, so callers
should project to the smallest edge field they need.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_edge_values tests.test_softmax tests.test_gat
DEV=METAL uv run python -m unittest tests.test_edge_values tests.test_softmax tests.test_gat
DEV=CPU uv run python -m experiments.gat
DEV=METAL uv run python -m experiments.gat
DEV=CPU uv run python -m unittest tests.test_multi_head_gat
DEV=METAL uv run python -m unittest tests.test_multi_head_gat
DEV=CPU uv run python -m experiments.multi_head_gat
DEV=METAL uv run python -m experiments.multi_head_gat
```
