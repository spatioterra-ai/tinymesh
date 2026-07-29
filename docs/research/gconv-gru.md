# GConvGRU experiment

This record asks what changes when hidden state, not only current input, moves
over the graph inside a recurrent cell.

## Decision

At tinygrad revision
[`7d48926`](https://github.com/tinygrad/tinygrad/tree/7d48926293484b78ea2f5a3c73108c3f52a36274),
Tinymesh matches an independent Chebyshev GConvGRU reference across two
snapshots on CPU and Metal.

At that revision the result added no public API. The checked contracts now live
in `tinymesh.nn.ChebConv` and `tinymesh.nn.GConvGRU`; this experiment retains
their host parity, sparse-call evidence, and controlled comparison.

## Chebyshev graph convolution

For an unweighted, loop-free, symmetric graph:

```text
S     = D^-1/2 A D^-1/2
L     = I - S
L_hat = 2L / 2 - I = -S
```

The final equality uses the same symmetric normalization and default
`lambda_max = 2` as PyTorch Geometric's
[ChebConv](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/nn/conv/cheb_conv.py#L71-L179).
Tinymesh rejects self-loops because that reference removes them before forming
the Laplacian. It rejects asymmetric connectivity because the current `Graph`
exposes incoming degree while this normalization needs one shared undirected
degree.

The Chebyshev basis is:

```text
T_0(X) = X
T_1(X) = L_hat X
T_k(X) = 2 L_hat T_(k-1)(X) - T_(k-2)(X)

Cheb_K(X) = Linear([T_0(X), T_1(X), ..., T_(K-1)(X)])
```

`L_hat X` is one existing sparse `Graph.sum` plus node-wise scaling. Order `K`
therefore uses `K - 1` sparse calls and stores `K` node fields. It never creates
an adjacency matrix or an edge-feature Cartesian product.

```text
X = T_0
|
+-- sparse shift ------------------> T_1
|                                    |
+-- subtract <--- 2 * sparse shift --+--> T_2
                                         |
                    repeat recurrence ---+--> ...
```

The independent host reference covers `K = 3`. A direct revision-bound
cross-check against the pinned PyG implementation agrees to float32 precision:

```text
PyG ChebConv   [-4.849998951, -3.799999952, -1.699999928]
Tinymesh       [-4.849999428, -3.799999714, -1.700000048]
```

## Cell

PyTorch Geometric Temporal's
[GConvGRU](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/gconv_gru.py#L36-L170)
uses separate Chebyshev convolutions for input and hidden state in each gate:

```text
Z       = sigmoid(Cheb_xz(X) + Cheb_hz(H))
R       = sigmoid(Cheb_xr(X) + Cheb_hr(H))
H_tilde = tanh(Cheb_xh(X) + Cheb_hh(R * H))
H_next  = Z * H + (1 - Z) * H_tilde
```

Chebyshev convolution is linear before its bias. Concatenating features fuses
each input-hidden pair without changing the function, and update and reset
share one input:

```text
[Z, R]  = sigmoid(Cheb_gates([X, H]))
H_tilde = tanh(Cheb_candidate([X, R * H]))
H_next  = Z * H + (1 - Z) * H_tilde
```

```text
[X_t, H_t-1] -------- Cheb_K -------- split --------> Z, R
     |                                                 |
     +---- [X_t, R * H_t-1] ---- Cheb_K ----> H_tilde |
                                                   |   |
H_t-1 ---------------------------------------------+---+--> H_t
```

For `K = 2`, this is two physical sparse calls per step rather than the six
logical convolutions in the reference layout. One fused bias replaces each
redundant pair of input and hidden biases; the represented functions are the
same, but the parameterization is smaller.

Mapping the checked `K = 3` fused weight blocks back into the six pinned PyG
Temporal convolutions gives:

```text
PyG Temporal   [0.154253155, -0.362862021, -0.027140588]
Tinymesh       [0.154253185, -0.362861991, -0.027140627]
maximum error   0.000000039
```

## Controlled comparison

T-GCN and GConvGRU use the same symmetric graph, two snapshots, hidden width
one, target, learning rate, and one full-model SGD step. Their initial
parameters represent the same function. The comparison reports unequal model
and execution cost instead of treating the cells as budget-matched:

```text
                              T-GCN       GConvGRU K=2
model parameters                 12                  15
sparse calls per step             1                   2
initial loss               0.655455            0.655455
spatial gradient           -0.170006             0.170006
loss after one step         0.138465            0.125983
```

The gradient signs differ because T-GCN uses normalized adjacency `S` while
the first non-constant Chebyshev term is `L_hat = -S`. Their equal magnitudes
at the aligned initialization confirm the mapping.

The lower final GConvGRU loss is not a model-quality result. GConvGRU has more
parameters, twice the sparse work at `K = 2`, and a different optimization
surface. This toy step proves gradients reach a Chebyshev spatial parameter
through time; it cannot rank architectures.

For input width `F`, hidden width `H`, and order `K`, the fused cells contain:

```text
T-GCN parameters       3FH + 6H^2 + 3H
GConvGRU parameters    3KH(F + H) + 3H

T-GCN sparse calls     1 per step
GConvGRU sparse calls  2(K - 1) per step
```

## What this adds

T-GCN graph-mixes the current input, then carries hidden state through
node-local gates. GConvGRU also graph-mixes hidden state:

```text
T-GCN       neighbors affect the current input path
GConvGRU    neighbors affect current input and remembered state
```

That is the architectural reason to test GConvGRU. It is not evidence that more
spatial recurrence will help a particular dataset.

The formulation follows
[ChebNet](https://arxiv.org/abs/1606.09375) and
[Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659).

## Limits

The result covers unweighted loop-free symmetric topology, fixed node identity,
regular ordered snapshots without interval metadata, one graph, one device,
zero initial hidden state, and first-order unrolling. It fixes
`lambda_max = 2` and supports no alternative Laplacian normalization.

It does not cover directed Chebyshev operators, edge weights, changing
topology, masks, irregular time, truncated backpropagation, long-horizon
stability, or production performance.

The cell now also accepts shared-graph batch axes. The
[Chickenpox forecast](chickenpox-forecast.md) records the first predictive
comparison without changing this controlled operator witness.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_gconv_gru
DEV=METAL uv run python -m unittest tests.test_gconv_gru
DEV=CPU uv run python -m experiments.gconv_gru
DEV=METAL uv run python -m experiments.gconv_gru
```
