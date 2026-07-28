# T-GCN experiment

This record asks whether one immutable `Graph` can carry a trainable recurrent
state across ordered node snapshots without a temporal kernel or container.

## Decision

At tinygrad revision
[`c9e1154`](https://github.com/tinygrad/tinygrad/tree/c9e11544df5db55c13f06d01fba5300dd44332fb),
Tinymesh unrolls a T-GCN cell across fixed-topology snapshots and differentiates
through space and time on CPU and Metal.

The result adds no public API and no `src/tinymesh` code. One graph is reused;
snapshots and hidden state are ordinary tinygrad tensors; the cell remains an
experiment.

## Why T-GCN first

PyTorch Geometric Temporal offers two adjacent gated designs:

- T-GCN runs three GCN operations over the current node field, then combines
  each result with node-local hidden state
  ([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/temporalgcn.py#L36-L130)).
- GConvGRU graph-convolves both input and hidden state for all three gates,
  requiring six Chebyshev convolutions per step
  ([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/gconv_gru.py#L55-L170)).

T-GCN is the smaller standard test of the temporal boundary because the
existing experimental GCN caller is sufficient. The subsequent
[GConvGRU experiment](gconv-gru.md) adds hidden-state propagation and compares
its parameter and sparse-call cost on the same temporal fixture.

The reference expresses its three input projections as separate GCN calls.
They share node input, topology, and normalization and differ only in output
weights, so linearity gives:

```text
[GCN_z(X), GCN_r(X), GCN_h(X)] = GCN_[W_z | W_r | W_h](X)
```

Tinymesh therefore makes one width-`3H` sparse call and slices its output into
the three gates. The model has the same three independent weight blocks without
repeating topology traversal.

## Cell

For current node field `X`, previous hidden state `H`, and graph convolution
`GCN_G`:

```text
Z       = sigmoid(Linear_z([GCN_G,z(X), H]))
R       = sigmoid(Linear_r([GCN_G,r(X), H]))
H_tilde = tanh(Linear_h([GCN_G,h(X), R * H]))
H_next  = Z * H + (1 - Z) * H_tilde
```

`Z` decides how much prior state survives. `R` decides how much prior state
enters the candidate. The candidate combines the current graph signal with
reset hidden state.

```text
X_t [N, F]
    |
 GCN [N, 3H]
    |
 split Z, R, candidate projections
    |              |              |
    +-- H_t-1      +-- H_t-1      +-- R * H_t-1
    |              |              |
 sigmoid Z      sigmoid R       tanh H_tilde
    |                             |
    +---------- gated update -----+
                  |
                  v
              H_t [N, H]
```

The graph caller uses `D^-1/2 A D^-1/2 XW`. Self-loops are explicit edges;
the cell does not silently change topology.

## Temporal data boundary

PyTorch Geometric Temporal's static signal container yields one ordinary graph
snapshot at a time while reusing one edge index
([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/signal/static_graph_temporal_signal.py#L14-L134)).

Tinymesh needs no container for this proof:

```text
Graph G                    one fixed topology and cache owner
snapshot X_t [N, F]        one ordered node field
hidden H_t [N, H]          one recurrent node field
```

The first call initializes hidden state to zero. Later calls validate its node
count, feature width, dtype, and device. Reusing the same `Graph` reuses its
lowered CSR buffers.

## Evidence

An independent host implementation evaluates all three gates over two
snapshots. Tinymesh matches it on CPU and Metal. Reversing the snapshots changes
the final state, which rejects order-insensitive aggregation. A UOp check sees
one `csr_sum` call in one cell step, enforcing the fused graph projection.

The learning witness is narrower:

```text
graph:       0 -> 0, 1 -> 1, 0 -> 1
snapshot 0: X_0 = [1, 0]
snapshot 1: X_1 = [0, 0]
target:      final node-1 state = 1
```

Node `1` receives the signal only through edge `0 -> 1` at the first snapshot.
The second snapshot contains no signal, so the prediction also requires
temporal retention. One SGD step updates the candidate graph parameter:

```text
initial loss                0.718740
candidate graph gradient   -0.188622
final loss                  0.686385
candidate graph weight      1.188622
```

Both backends return the same float32 values. This proves a parameter gradient
crosses one spatial edge and one temporal transition. It does not establish
forecast quality.

The formulation follows
[T-GCN](https://arxiv.org/abs/1811.05320).

## Limits

The result covers one fixed graph, ordered snapshots without interval metadata,
zero initial hidden state, explicit first-order unrolling, and one device. It
does not cover timestamps, irregular intervals, missingness masks, changing
edge values, changing topology, batches, state detachment, truncated
backpropagation, multi-step losses, long-horizon stability, or useful predictive
accuracy.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_tgcn
DEV=METAL uv run python -m unittest tests.test_tgcn
DEV=CPU uv run python -m experiments.tgcn
DEV=METAL uv run python -m experiments.tgcn
```
