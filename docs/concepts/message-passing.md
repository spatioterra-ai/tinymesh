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
message:    m(u -> v) = x_u
aggregate:  a_v       = sum(m(u -> v)) / max(1, d_v)
update:     y_v       = W_neighbor a_v + b_neighbor + W_root x_v
```

`d_v` is the incoming degree of node `v`. `Graph.mean` returns zero for an
isolated node. `SAGEConv` then applies its neighbor linear map, so an enabled
neighbor bias also reaches isolated nodes.

A shared bias-free linear map commutes with mean. The original experiment put
that map before aggregation; the public class aggregates first so its optional
bias is applied exactly once after the mean. The checked bias-free witness has
the same values and parameter gradient under either factorization.

## GCN composition

The second caller uses the same sum with symmetric degree normalization:

```text
message:    m(u -> v) = x_u / sqrt(d_u)
aggregate:  a_v       = sum(m(u -> v) for each edge u -> v)
update:     y_v       = W(a_v / sqrt(d_v)) + b
```

For unit edges this is `(D^-1/2 A D^-1/2 X)W + b`. A bias-free shared linear
map commutes with the normalized sum, so this is the same equation as
`D^-1/2 A D^-1/2 XW`. The public class aggregates at input width and applies
the optional bias last. Source and destination normalization are ordinary
node-wise tensor operations, so they compose around the existing CSR sum. The
caller includes self-loops explicitly. Integer degree is a topology fact;
reciprocal and inverse square root are model semantics.

## Scalar weighted messages

The first edge-dependent message multiplies each source row by one scalar:

```text
message:    m(u -> v, e) = w_e x_u
aggregate:  y_v          = sum(w_e x_u for each edge e: u -> v)
```

For incoming output gradient `g_v`, reverse mode needs:

```text
dX_u = sum(w_e g_v for each edge e: u -> v)
dw_e = dot(x_u, g_v)
```

`dX` is another weighted CSR sum over the transpose. Each `dw_e` has exactly
one edge owner and reduces only feature width `H`, so it needs neither atomics
nor a materialized `[E, H]` message tensor. Edge-order maps keep the scalar
attached to the same COO edge through both traversals.

## Graph attention

A single attention head composes node transforms, endpoint projection, target
softmax, and weighted sum:

```text
z_v       = W x_v
score_uv  = LeakyReLU(dot(a_source, z_u) + dot(a_target, z_v))
alpha_uv  = softmax(score_uv over every edge ending at v)
y_v       = sum(alpha_uv z_u for every edge u -> v)
```

`Graph.edge_values` projects node coefficients to source or target positions in
COO edge order. `Graph.softmax` normalizes scalar edge scores among each
target's incoming edges. `Graph.sum` then consumes those coefficients as scalar
edge weights. Ordinary tinygrad operations own every learned transform.

Multiple heads add a parameter axis, not a new graph operation:

```text
state       [N, K, C]
node score  [N, K]
edge score  [E, K]
    |
    +--> head 0: target softmax -> weighted sum [N, C] --+
    +--> head 1: target softmax -> weighted sum [N, C] --+--> concatenate
    `--> ...                                               [N, K*C]
```

Each head normalizes only against other edges in that head. The current
experiment calls scalar `Graph.softmax` and `Graph.sum` once per head. Kernel
count therefore grows with head count, but topology and semantics stay shared.

## Temporal recurrence

T-GCN places three logical GCN projections inside a GRU:

```text
current X_t -- GCN --> update, reset, candidate gates
previous H_t-1 -----> update, reset, candidate gates
                                |
                                v
                              H_t
```

The three projections share input, topology, and normalization, so Tinymesh
performs one sparse reduction at input width `F`, applies one linear map to
`3H` channels, and then slices them. The gates retain or replace node-local
hidden state. Calling the same cell over an ordered tensor sequence creates a
gradient path through both graph propagation and earlier hidden state. Read
[Time](time.md) for the temporal data contract.

GConvGRU moves hidden state over the graph too:

```text
[Z, R]  = sigmoid(Cheb_K([X, H]))
H_tilde = tanh(Cheb_K([X, R * H]))
```

The Chebyshev basis repeatedly applies a scaled normalized adjacency through
the same sparse sum. Order `K` uses `K - 1` sparse calls per convolution. Input
and hidden projections fuse along feature width, so the recurrent cell needs
two Chebyshev convolutions rather than six separate logical ones. This changes
model capacity and cost; it does not change `Graph`.

## Gradient path

The layer's forward flow is:

```text
X -- A @ X -- inverse degree -- W_neighbor --+
                                             +--> Y
X ----------------------------- W_root ------+
```

Reverse-mode differentiation follows the graph in reverse:

```text
dY
 |--> gradient of W_root -------------------------------> dX
 +--> gradient of W_neighbor --> W_neighbor.T
                                  |
                                  +--> inverse degree --> A.T @ gradient --> dX
```

The transpose CSR is not a second graph algorithm. It lets the same sparse sum
return destination gradients to the source states that produced their
messages. The layer-parameter gradient consumes the sparse aggregate directly;
the value gradient traverses transpose CSR.

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

That proves first-order parameter learning from the checked-in sparse
aggregate. Separate gradient tests prove the transpose-CSR path, and the CSR
record proves its scaling structure. The witness does not prove
generalization, useful model quality, depth, temporal learning, or training on
a real graph.

## Where other layers differ

The same decomposition identifies capabilities that do not exist yet:

- GIN keeps sum aggregation and adds a learned update.
- Edge-conditioned attention adds edge features to score construction.

GIN and edge-conditioned attention remain design probes. Mean GraphSAGE, GCN,
single- and multi-head GAT, T-GCN, and GConvGRU now compose as direct
`tinymesh.nn` classes over the shared experimental `Graph` boundary. The
classes own equations and parameters; experiments own training, controls, and
claims.

The general formulation follows
[GraphSAGE](https://arxiv.org/abs/1706.02216). The exact learning result lives
in [Mean GraphSAGE experiment](../research/mean-sage.md). The normalized second
caller lives in [GCN experiment](../research/gcn.md). Scalar edge evidence
lives in
[Weighted aggregation experiment](../research/weighted-aggregation.md).
Endpoint projection, segment softmax, and the attention witness live in
[Sparse attention experiment](../research/attention.md).
Fixed-topology recurrence lives in
[T-GCN experiment](../research/tgcn.md) and
[GConvGRU experiment](../research/gconv-gru.md).
