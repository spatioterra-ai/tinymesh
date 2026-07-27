# GCN experiment

This record asks whether a second graph layer needs a larger sparse primitive
than the mean-GraphSAGE experiment.

## Decision

At tinygrad revision
[`bdbb1d7`](https://github.com/tinygrad/tinygrad/tree/bdbb1d702f91c68ccfb0b93d93180b6f0947c7c1),
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

GCN therefore does not yet justify an edge-weight primitive. Topology owns
connectivity; the layer owns normalization and its linear map. The caller
supplies the self-loops required by the GCN renormalization rule, keeping
topology mutation explicit.

This is the second semantic caller for CSR sum, not evidence for a stable
runtime API. The implementation still crosses tinygrad's alpha
`Tensor.custom_kernel` boundary and disables default kernel optimization.

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

The experiment covers unweighted, fixed topology and first-order gradients.
Topology caches integer degree, while each call derives inverse square root
with ordinary tinygrad operations. Weighted edges, implicit self-loop insertion,
batching, depth, nonlinearities, and model quality remain unproven. In
particular, a weighted GCN may require edge-local multiplication and must earn
that boundary with its own sparse evidence.

The formulation follows Kipf and Welling's
[paper](https://arxiv.org/abs/1609.02907); the
[PDF](https://arxiv.org/pdf/1609.02907) and
[TeX source](https://arxiv.org/src/1609.02907) are available from arXiv.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_gcn
DEV=METAL uv run python -m unittest tests.test_gcn
DEV=CPU uv run python -m experiments.gcn
DEV=METAL uv run python -m experiments.gcn
```
