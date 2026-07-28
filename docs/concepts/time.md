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

## Resolution belongs to the data

The cell sees order, not calendar meaning. Hourly, monthly, and quarterly
snapshots use the same recurrence, but they are not interchangeable evidence.
An application must retain the timestamp or interval represented by every
position and must define how irregular gaps affect features and loss.

Missing state is not numeric zero. A future temporal data contract needs
explicit masks when observations may be absent.

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
and yields `(x_t, y_t)` in order. Contiguous temporal splits reuse the same
graph and edge facts:

```python
train, test = signal.split(0.8)
for x, y in train:
    hidden = cell(x, train.graph, hidden)
```

The container does not claim more than its source. The current public dataset
has ordered weekly positions but no exact dates or missingness mask, so those
are not fabricated. Read the
[Chickenpox data record](../research/chickenpox-data.md) for the concrete
caller and parity evidence.

## Where spatial mixing happens

Two fixed-graph cells now expose one architectural choice:

```text
T-GCN       graph-mix X_t, then combine node-local H_(t-1)
GConvGRU    graph-mix X_t and H_(t-1) inside the gates
```

T-GCN is cheaper. GConvGRU lets a node's remembered state affect neighboring
nodes before the next update. Whether that extra path helps is a model and data
question, not a data-container question. Both consume the same explicit
`Graph`, snapshots, and hidden tensor.

The exact checked result lives in the
[T-GCN experiment](../research/tgcn.md). The controlled architectural
comparison lives in the [GConvGRU experiment](../research/gconv-gru.md).
