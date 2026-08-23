# Time

Time changes tensor fields before it changes topology.

The smallest temporal graph contract keeps one graph `G` and presents an
ordered sequence of node fields:

```text
fixed topology G
      |
      +-------+-------+-------+
      |       |       |       |
     X0      X1      X2      ...
      |       |       |
      v       v       v
    Cell -> Cell -> Cell
      |       |       |
     H0      H1      H2
```

Each `X_t` has shape `[N, F]`. Each hidden state `H_t` has shape `[N, H]`.
Node identity and row order stay stable across the sequence.

## Batch lanes share one graph

A batch of windows adds leading axes; it does not copy topology:

```text
values [B, L, N, H]
             |
             | node axis stays owned by Graph
             v
       [N, B * L * H]
             |
             | one destination-CSR sum
             v
       [N, B * L * H]
             |
             v
output [B, L, N, H]
```

`Graph.sum` accepts `[..., N, H]`, moves `N` first, folds every independent
lane into feature width, runs the existing two-dimensional CSR operation once,
and restores the original axes. The graph and its device buffers remain shared.
Static scalar edge weights are also shared; their gradient sums over all lanes.
`Graph.edge_values` follows the same rule, so `GINEConv` can share one fixed
edge-feature tensor across batch and time lanes.

This is batching tensor fields over one graph. Batching different graphs,
changing edge values per batch, and batched attention scores are separate
contracts and remain unimplemented.

## Recurrence is causal

One step is:

```text
H_t = Cell(X_t, G, H_(t-1))
```

`H_t` may depend on the current and earlier snapshots, never a later one.
Unrolling ordinary tinygrad calls across an explicit Python sequence builds one
autograd graph through both space and time.

Training and evaluation splits must preserve that direction. A feature attached
to snapshot `t` must have been observable by `t`; putting future facts into an
earlier row is leakage even if the recurrent code is causal.

## Windows make history explicit

`StaticGraphTemporalSignal.batches()` converts aligned snapshots into causal
sequence-to-one windows:

```text
x [T, N, F] -- history L --> values [B, L, N, F]
y [T, N, Y] ----------------> target [B, N, Y]

values[b] = x[start:start + L]
target[b] = y[start + L - 1]
```

The last target aligns with the last input snapshot's declared label. For the
Chickenpox loader with `lags=1`, that label is the following week's value.
Splitting the signal before creating windows prevents any window from crossing
a split boundary.

The iterator keeps the final short batch. It creates each window batch from
`L` contiguous tensor slices and never duplicates topology.

## Resolution belongs to the data

The cell sees order, not calendar meaning. Hourly, monthly, and quarterly
snapshots use the same recurrence, but they are not interchangeable evidence.
An application must retain the timestamp or interval represented by every
position and must define how irregular gaps affect features and loss.

Missing state is not numeric zero. A future temporal data contract needs
explicit masks when observations may be absent. METR-LA is one concrete caller:
its reference protocol declares zero a missing-value sentinel, so
`METRLA.observed` exposes `speed != 0` while retaining the raw reading tensor.

## Fixed and changing graphs differ

Three cases should remain distinct:

```text
fixed topology       G, X_t
changing edge facts  G, X_t, w_t
changing topology    G_t, X_t
```

The first case reuses one lowered graph and its device buffers. The second keeps
edge identity but changes aligned values. The third changes connectivity and
requires explicit rules for node identity, graph versions, hidden-state
alignment, and cache lifetime.

Proving fixed-topology recurrence does not prove dynamic graphs.

## A node-time mesh is a graph product

A fixed graph observed at ordered times has a conceptual joint domain:

```text
J = G x T

joint node       (v, t)
spatial edge     (u, t)   -> (v, t)
temporal edge    (v, t-1) -> (v, t)
```

The [time-vertex framework](https://arxiv.org/abs/1705.02307) writes its joint
Laplacian as a Kronecker sum:

```text
L_J = L_T ⊗ I_N + I_T ⊗ L_G
```

The default execution stays factorized: keep `X[T,N,F]`, apply the spatial
operator over `N`, and advance causal state over `T`:

```text
for t:
  H[t] = Cell(X[t], G, H[t-1])
```

Topology and spatial transport remain proportional to the snapshots and sparse
support, `O(T * (N + E) * H)`; learned feature projections add their ordinary
tensor cost. A changing edge field, delayed cross-time edge, or changing
topology needs its own aligned contract.

Some algorithms need the joint nodes to exchange messages directly. For a
bounded window, `Graph.cartesian` lowers that same product without constructing
a dense adjacency:

```python
time = Graph(3, [0, 1], [1, 2])
space = Graph(2, [0, 1], [1, 0])
mesh = time.cartesian(space)

values = values.reshape(batch, time.nodes * space.nodes, features)
```

```text
flat node       (t, v) -> t * N + v
joint nodes     T * N
joint edges     E_T * N + T * E_G

time edge       t -> u    becomes (t, v) -> (u, v) for every v
space edge      v -> w    becomes (t, v) -> (t, w) for every t
```

Left-factor edges come first in COO order, each repeated across the right
factor's nodes; right-factor edges follow, each repeated across the left
factor's nodes. Callers can therefore align edge types and values without a
second topology map. For `time.cartesian(space)`, temporal edges precede
spatial edges. This order belongs to that binary bracketing: regrouping three or
more factors preserves the directed edge multiset but can permute COO order, so
aligned edge values must be derived for the selected bracketing.

This explicit form costs `O(TN + E_T N + T E_G)` storage. Use it when a joint
message-passing rule needs it and keep long fixed-topology sequences
factorized. The Cartesian product is a lowering choice, not permission to
materialize an `[TN,TN]` matrix.

## The fixed-graph signal

The first real dataset caller adds invariants that a tuple cannot carry:

```text
Graph G
node IDs [N]
x [T, N, F]
y [T, N, Y]
optional edge weight [E]
```

`StaticGraphTemporalSignal` validates those axes, keeps topology in one owner,
and yields `(x_t, y_t)` in order or batched windows on request. Contiguous
temporal splits reuse the same graph and edge facts:

```python
train, test = signal.split(0.8)
for x, y in train:
    hidden = cell(x, train.graph, hidden)

values, target = next(train.batches(batch_size=32, history=8))
# values [B, L, N, F], target [B, N, Y]
```

The container does not claim more than its source. Chickenpox has ordered
weekly positions but no exact dates or missingness mask, so those are not
fabricated. METR-LA instead retains its exact timestamps and source-defined
missingness in a dataset-specific record; that evidence is not yet broad
enough to enlarge `StaticGraphTemporalSignal`. Read the
[Chickenpox data record](../research/chickenpox-data.md) and
[METR-LA data record](../research/metr-la-data.md) for the concrete boundaries.

## Where spatial mixing happens

The fixed-graph temporal components expose three architectural choices:

```text
T-GCN       graph-mix X_t, then combine node-local H_(t-1)
PeriodAttention
            mix P same-shaped states; no graph assumption
A3T-GCN     T-GCN each period from fixed H_0, then PeriodAttention
GConvGRU    graph-mix X_t and H_(t-1) inside the gates
```

`PeriodAttention` is the composable temporal primitive: it can mix states from
any encoder without knowing whether they came from a graph, spectral filter, or
future multiscale operator. A3T-GCN mixes independent T-GCN period encodings
rather than carrying hidden state through them. GConvGRU lets a node's
remembered state affect neighboring nodes before the next update. Whether
either extra path helps is a model and data question, not a data-container
question. The graph cells consume the same explicit `Graph`, snapshots, and
hidden tensor.

The exact checked result lives in the
[T-GCN experiment](../research/tgcn.md). The controlled architectural
comparison lives in the [GConvGRU experiment](../research/gconv-gru.md). The
first real batched forecast lives in the
[Chickenpox forecast](../research/chickenpox-forecast.md).
