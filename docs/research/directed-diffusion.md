# Directed diffusion experiment

This experiment asks whether the existing sparse graph operations can express
source-normalized propagation in both directions of one directed graph.

## Decision

At tinygrad revision
[`dd16d5a`](https://github.com/tinygrad/tinygrad/tree/dd16d5aead62e0207c0c3c50c19bc8b67e176c55),
Tinymesh composes bidirectional diffusion without a new kernel or public API:

```text
positive affinity[E]
       |
       +--> outgoing sum by source --> forward weight[E] --> G.sum
       |
       +--> outgoing sum in G.T ----> reverse weight[E] --> G.T.sum
```

`DirectedDiffusion` remains under `experiments/`. It owns the reverse graph and
two normalized edge fields. A fixed recurrent caller realizes those fields
once before reuse.

## Operator

For every original edge `e: u -> v`:

```text
p_forward[e] = a[e] / sum(a[k] for k with source[k] = u)
p_reverse[e] = a[e] / sum(a[k] for k with target[k] = v)

forward(X)[v] += p_forward[e] X[u]
reverse(X)[u] += p_reverse[e] X[v]
```

The second denominator is the outgoing degree of `v` in the reversed graph.
The reverse `Graph(nodes, target, source)` keeps the original COO edge order,
so both normalized fields preserve one identity.

Each edge has a positive affinity and therefore belongs to a row with a
positive denominator. Isolated nodes own no edge, require no division
convention, and return zero.

## Composition

The current public operations are sufficient:

```text
ones[N,1]
    |
    +--> G.T.sum(ones, a) --> outgoing[N,1]
    |                              |
    |                       edge_values(source)
    |                              |
    |                       a / degree --> p_forward[E]
    |
    +--> G.sum(ones, a) ---> incoming[N,1]
                                   |
                            edge_values(target)
                                   |
                            a / degree --> p_reverse[E]
```

Preparing the fixed weights performs two sparse sums and two endpoint
projections once. Realizing them marks the boundary between differentiable
affinity construction and a fixed recurrent cache. Each later application
performs exactly two sparse sums. Leading batch axes reuse the same topology
and weights.

## Exact witness

An independent Python edge loop computes both normalized fields, both
directions, and gradients with respect to node values and raw affinity. The
Tinygrad result matches it within `1e-5` on CPU and Metal.

The fixture contains asymmetric degrees, duplicate edges, and two isolated
nodes. Reordering COO edges reorders the two weight fields and affinity
gradients while preserving node outputs and node gradients.

## Sparse work

After preparation, the only model-time intermediates are the two `[N,H]`
outputs and fixed `[E]` weight fields:

```text
forward       one csr_sum
reverse       one csr_sum
topology      O(N + E)
node fields   O(NH)
edge fields   O(E)
```

UOp inspection rejects `[N,N]` and `[N,E]` carriers. This is a structural
complexity claim, not a performance claim.

## Reference boundary

The operator follows the forward and reverse random-walk idea in
[DCRNN](https://arxiv.org/abs/1707.01926). The pinned PyG Temporal
implementation constructs a dense adjacency before propagation. Tinymesh
keeps both directions sparse and removes the two redundant learned local
blocks; this experiment does not claim framework parity.

## Limits

The result covers fixed directed topology, positive scalar affinity, shared
graph batch axes, first-order gradients, and one-step diffusion. It does not
cover learned affinity, zero or negative affinity, higher-order walks,
changing topology, vector edge messages, recurrent model quality, or a public
diffusion API.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_directed_diffusion
DEV=METAL uv run python -m unittest tests.test_directed_diffusion
DEV=CPU uv run python -m experiments.directed_diffusion
DEV=METAL uv run python -m experiments.directed_diffusion
```
