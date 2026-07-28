# Sparse graph topology

A graph gives relationships to tensor values. tinymesh currently starts with a
directed graph:

```text
node state: X[N, H]
edge:       source -> target
```

`N` is the number of nodes and `H` is the feature width. Topology says which
rows of `X` may interact; it does not contain the node values themselves.
Public `Graph` owns ordered COO identity; its replaceable private backend owns
CSR lowering and device caches.

## COO: edges as pairs

Coordinate (COO) format stores one source and target per edge:

```python
source = [0, 1, 1]
target = [2, 2, 3]
```

This means:

```text
0 ---> 2
1 ---> 2
1 ---> 3
```

COO is convenient at the boundary because edges are explicit. It is not the
current execution form: finding every incoming edge for one destination would
otherwise require searching the whole list.

## CSR: edges grouped by output owner

Compressed sparse row groups incoming sources by destination:

```text
destination  incoming sources
0            []
1            []
2            [0, 1]
3            [1]
```

Two arrays encode those rows:

```python
row_ptr = [0, 0, 0, 2, 3]
column  = [0, 1, 1]
```

The sources for node `v` occupy:

```python
column[row_ptr[v]:row_ptr[v + 1]]
```

tinymesh sorts each row by source. Input edge order therefore does not change
CSR connectivity for a fixed node labeling. A separate edge-order map retains
which COO edge produced each CSR position. Duplicate edges remain separate and
keep their identities and multiplicity; empty rows remain explicit.

## Adjacency without a dense matrix

It is useful to describe propagation with an adjacency matrix `A`:

```text
A[v, u] = number of edges u -> v
Y       = A @ X
```

This is notation for a linear map, not the implementation. tinymesh does not
construct an `N x N` matrix and does not call dense matrix multiplication. The
CSR row for `v` directly visits the stored sources and sums their features.

## Why store the transpose?

For fixed topology:

```text
Y = A @ X
```

first-order reverse-mode differentiation needs:

```text
dX = A.T @ dY
```

`A.T` reverses ownership: each source row lists the targets that consumed it.
The same CSR sum can therefore implement forward and backward:

```text
forward CSR:    destination row -> source columns -> sum source features
transpose CSR:  source row      -> target columns -> sum target gradients
```

One primitive carries both directions. The two CSR forms contain
`2E + 2(N + 1)` integers.

## Edge identity survives lowering

Connectivity is not the only edge fact. A weight, distance, timestamp, or
provenance value must stay attached to the source-target pair it describes.

For one unordered COO input:

```text
COO edge ordinal       0       1       2
edge                  1 -> 3  0 -> 2  1 -> 2
weight                -1.0     0.5     2.0

CSR position           0       1       2
edge                  0 -> 2  1 -> 2  1 -> 3
edge_order              1       2       0
weight lookup          w[1]    w[2]    w[0]
```

`edge_order[p]` is the original COO ordinal for forward CSR position `p`.
The weighted kernel uses that private map to read the caller's tensor without
reordering it.

Backward uses the same identity:

```text
transpose_order[transpose position] -> original COO ordinal
dw[e] = dot(X[source[e]], dY[target[e]])
```

The transpose map selects the right raw weight while computing `dX`. The
edge-gradient kernel gives each original COO edge one output, so `dw` has the
same order as the input weight tensor. Parallel edges keep distinct ordinals.

The topology retains original `source` and `target` tuples plus the forward and
transpose order maps. Weighted execution realizes those `4E` identity
ordinals. Scalar weights add `E` values. Storage remains `O(N + E)`; no
`[E, H]` edge-feature product is materialized.

## Pull versus push

A push kernel gives work to edges:

```text
for each source -> target:
    output[target] += input[source]
```

Many edges may write the same destination, so parallel execution needs atomic
addition or another reduction step.

The current pull kernel gives each output scalar one owner:

```text
for each destination and feature:
    visit its CSR row
    write one sum
```

No two workers write the same output. The cost is degree skew: a high-degree hub
must traverse a long row serially for each feature.

## Lowering and reuse

Lowering changes representation while preserving meaning:

```text
COO edge facts
    | deterministic grouping
    v
Python CSR tuples + private order maps
    | device realization
    v
tinygrad integer tensors
```

The edge list remains the semantic source of truth. CSR is the execution form;
edge tensors remain in COO order and are selected through the private maps.
Fixed topology can reuse its realized execution form across training steps;
cache ownership and measurements belong to the revision-bound research
records.

The current sparse invariant is:

> Network-scale graph computation may store or visit node and edge lanes, but
> must not materialize node-pair or node-edge products.

The revision-bound implementation and measurements live in
[Sparse aggregation feasibility](../research/sparse-aggregation.md) and
[Weighted aggregation experiment](../research/weighted-aggregation.md).
