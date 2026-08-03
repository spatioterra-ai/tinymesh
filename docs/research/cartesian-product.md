# Sparse Cartesian products

A bounded node-time mesh can be one ordinary sparse graph. The Cartesian
product composes domains without adding a temporal graph type, dense adjacency,
or runtime dependency.

```text
time P_T  □  space G
       |
       +-- node (t, v) -> t * N + v
       +-- time edges  -> same v, adjacent t
       +-- space edges -> same t, edge in G
       |
       v
Graph(T * N, E_T * N + T * E_G)
```

`Graph.cartesian` uses left-major node identity. It emits every left-factor
edge across the right nodes first, then every right-factor edge across the left
nodes. A caller can reshape `[B,T,N,H]` to `[B,T*N,H]` and derive aligned
temporal and spatial edge values from that order.

This is an explicit lowering for algorithms that message across joint nodes.
Fixed-topology recurrence remains the smaller representation for long
sequences: it keeps one `G` instead of repeating its edges for every time in a
window. Neither path constructs `[TN,TN]` state.

## Decision

At revision
[`ccee5d9`](https://github.com/spatioterra-ai/tinymesh/tree/ccee5d9a208944ba7b634398b614969b1d257afe),
30 focused tests pass identically on CPU and Metal.

- A three-step path and two-node directed graph lower to six left-major nodes
  and ten edges, exactly `E_T*N + T*E_G`.
- Batched endpoint gathering preserves COO order and its first-order gradient
  through one sparse gather and one sparse transpose sum.
- Batched `GINEConv` agrees with independent lane calls within `1e-6` while
  sharing one topology and one edge-feature tensor.

Promote `Graph.cartesian`, batched `Graph.edge_values`, and batched `GINEConv`.
They extend existing owners with standard shape and ordering contracts. Product
edge values per lane, changing products, and unbounded sequence expansion stay
out of scope.

## Reproduce

```console
DEV=CPU uv run --locked python -m unittest tests.test_graph tests.test_edge_values tests.test_gine
DEV=METAL uv run --locked python -m unittest tests.test_graph tests.test_edge_values tests.test_gine
```
