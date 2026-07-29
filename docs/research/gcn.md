# GCN experiment

This record asks whether a second graph layer needs a larger sparse primitive
than the mean-GraphSAGE experiment.

## Decision

At tinygrad revision
[`0bb36c9`](https://github.com/tinygrad/tinygrad/tree/0bb36c99899ba4742dbe1d5784397373998d81c3),
an unweighted GCN layer composes from the same destination-CSR sum on CPU and
Metal:

```text
X -- linear --> XW -- source scale --> A @ (...) -- destination scale --> Y
```

For unit edges, symmetric normalization factorizes into one source-node scale
and one destination-node scale:

```text
Y = D^-1/2 A D^-1/2 XW
```

GCN therefore does not require an edge-weight primitive. Topology owns
connectivity; the layer owns normalization and its linear map. A later
[scalar-weight experiment](weighted-aggregation.md) earns edge-local
multiplication independently. The caller supplies the self-loops required by
the GCN renormalization rule, keeping topology mutation explicit.

This was the second semantic caller for `Graph.sum`, not model-quality
evidence. The proven equation now lives in `tinymesh.nn.GCNConv`; the
experiment retains its dense reference and learning witness. The private graph
implementation still crosses tinygrad's alpha `Tensor.custom_kernel` boundary
with default kernel optimization disabled.

The public class commutes the shared linear map after normalized aggregation
and applies its optional bias last:

```text
X -- source scale --> A @ (...) -- destination scale --> linear --> Y
```

The recorded witness disables bias, so this factorization has the same values
and weight gradient as the historical path. Its sparse call carries input
feature width rather than output feature width.

## Evidence

A three-node path with self-loops has degrees `2, 3, 2`. The experiment matches
a host-computed dense reference, so the check exercises unequal source and
destination normalization rather than a regular-graph shortcut. A separate
fixture verifies vertex-permutation equivariance and an explicit self-loop on a
disconnected node; a zero-degree row returns zero without dividing by zero.

The learning witness uses two disjoint two-node components:

```text
features:  x_0 = 1, x_1 = -1, x_2 = 0, x_3 = 0
edges:     0 <-> 2, 1 <-> 3, plus one self-loop per node
targets:   y_2 = 1, y_3 = -1
```

Nodes `2` and `3` have identical root features. Normalized propagation gives
them `0.5W` and `-0.5W`. Starting `W` at zero gives loss `1` and gradient `-1`;
one SGD step at learning rate `2` sets `W = 2` and reaches loss `0`. CPU and
Metal reproduce those values to float32 precision.

## Limits

The experiment covers unit edges, fixed topology, and first-order gradients.
Topology derives integer degree from cached row pointers, while each call
derives inverse square root with ordinary tinygrad operations. Scalar weighted
sum is proven separately, but weighted GCN normalization semantics are not
selected. Implicit self-loop insertion, batching, depth, nonlinearities, and
model quality remain unproven.

The formulation follows Kipf and Welling's
[paper](https://arxiv.org/abs/1609.02907); the
[PDF](https://arxiv.org/pdf/1609.02907) and
[TeX source](https://arxiv.org/src/1609.02907) are available from arXiv.

## Reproduce

```console
DEV=CPU uv run --locked python -m unittest tests.test_gcn
DEV=METAL uv run --locked python -m unittest tests.test_gcn
uv run --locked python -m experiments.run gcn DEV=CPU
uv run --locked python -m experiments.run gcn DEV=METAL
```
