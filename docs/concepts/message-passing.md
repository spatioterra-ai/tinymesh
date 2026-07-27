# Message passing

Message passing turns graph structure into tensor computation. One layer has
three semantic steps:

```text
message:    m(u -> v) = M(x_u, x_v, e_uv)
aggregate:  a_v       = Aggregate(m(u -> v) for each edge u -> v)
update:     y_v       = U(x_v, a_v)
```

- `M` computes information carried by one edge.
- `Aggregate` combines a destination's incoming messages.
- `U` combines neighborhood information with the destination's own state.

`x_u` and `x_v` are source and destination state; `e_uv` is optional edge
state.

The separation matters because topology owns connectivity while ordinary
tinygrad tensor operations can own most model-specific transformations.

## Current mean GraphSAGE composition

tinymesh's first caller uses:

```text
message:    m(u -> v) = W_neighbor x_u
aggregate:  a_v       = sum(m(u -> v)) / max(1, d_v)
update:     y_v       = W_root x_v + a_v
```

`d_v` is the incoming degree of node `v`. An isolated node receives zero
neighborhood state and keeps only its root path.

The neighbor transform runs before aggregation. For a bias-free linear map,
transforming before or after a mean gives the same forward value, but placing it
before aggregation proves that its parameter gradient crosses the sparse
backward boundary.

## GCN composition

The second caller uses the same sum with symmetric degree normalization:

```text
message:    m(u -> v) = W x_u / sqrt(d_u)
aggregate:  a_v       = sum(m(u -> v) for each edge u -> v)
update:     y_v       = a_v / sqrt(d_v)
```

For unit edges this is `D^-1/2 A D^-1/2 XW`. Source and destination
normalization are ordinary node-wise tensor operations, so they compose around
the existing CSR sum. The experiment requires the caller to include self-loops
explicitly.

## Gradient path

The layer's forward flow is:

```text
X -- W_neighbor --> messages -- A @ messages -- inverse degree --+
                                                                 +--> Y
X -- W_root -----------------------------------------------------+
```

Reverse-mode differentiation follows the graph in reverse:

```text
dY
 |--> gradient of W_root
 +--> inverse degree --> A.T @ gradient --> gradient of W_neighbor
```

The transpose CSR is not a second graph algorithm. It lets the same sparse sum
return destination gradients to the source states that produced their
messages.

## What one witness proves

The checked-in experiment uses:

```text
features:  x_0 = 1, x_1 = -1, x_2 = 0, x_3 = 0
edges:     0 -> 2, 1 -> 3
targets:   y_2 = 1, y_3 = -1
```

Nodes `2` and `3` have identical root features, so a root-only function cannot
distinguish them. Neighbor information can. Starting both linear weights at
zero gives loss `1` and neighbor gradient `-2`; one SGD step sets the neighbor
weight to `1` and reaches loss `0`.

That proves first-order parameter learning through the checked-in sparse
boundary. The separate CSR record proves its scaling structure. The witness
does not prove generalization, useful model quality, depth, temporal learning,
or training on a real graph.

## Where other layers differ

The same decomposition identifies capabilities that do not exist yet:

- GIN keeps sum aggregation and adds a learned update.
- Attention requires edge scores and a sparse normalization such as segment
  softmax.
- Temporal graph models repeat or evolve spatial state across ordered
  snapshots.

These remain design probes, not supported architectures. Mean GraphSAGE and
GCN now expose a shared semantic boundary, but the alpha execution path still
blocks a stable public composition.

The general formulation follows
[GraphSAGE](https://arxiv.org/abs/1706.02216). The exact learning result lives
in [Mean GraphSAGE experiment](../research/mean-sage.md). The normalized second
caller lives in [GCN experiment](../research/gcn.md).
