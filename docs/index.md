# tinymesh

tinymesh is a tinygrad-native library for learning over sparse structures
through space and time.

A graph is the smallest mesh. Nodes carry tensor fields and edges say which
nodes may interact. Coordinates, higher-dimensional cells, and time can extend
that structure without replacing the sparse core.

tinymesh is experimental. It currently proves fixed-graph sparse aggregation
and one trainable GraphSAGE layer on CPU and Metal. It does not expose a stable
public API yet.

## Try it

From a repository checkout:

```console
uv sync --locked
```

```python
from tinygrad import Tensor

from experiments.csr_aggregation import CSRTopology, csr_edge_sum

topology = CSRTopology(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

print(csr_edge_sum(state, topology).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Node `2` receives values from nodes `0` and `1`. Node `3` receives the value
from node `1`. The other nodes have no incoming edges, so their aggregate is
zero.

The [quick start](quickstart.md) follows this value through topology lowering,
sparse execution, backward propagation, and one trainable layer.

## The stack

```text
edge facts             source -> target
    |
    v
topology lowering      COO -> CSR(A) + CSR(A.T)
    |
    v
sparse operation       A @ X forward, A.T @ dY backward
    |
    v
model composition      mean GraphSAGE
```

[tinygrad](https://github.com/tinygrad/tinygrad) owns tensors, autograd,
compilation, and device execution. tinymesh owns sparse topology, mesh
semantics, and the model compositions that need them. It is not a PyTorch
Geometric compatibility layer.

The current operation stores `O(N + E)` topology and performs
`O((N + E)H)` work for `N` nodes, `E` edges, and feature width `H`. It never
constructs dense node-pair or node-edge state.

## What works

- deterministic directed edge-list to CSR lowering;
- destination-owned sparse sum with no atomic writes;
- the same operation over transpose CSR for first-order backward;
- fixed-topology device-buffer reuse;
- one mean-GraphSAGE composition whose neighbor parameter learns through the
  sparse boundary;
- CPU and Metal tests.

The implementation remains under `experiments/`. Its custom kernel uses an
alpha tinygrad surface and disables default kernel optimization for its
data-dependent CSR loop. Weighted messages, attention, batching, changing
topology, higher-order gradients, geometry, cells, and time are not implemented.

## Learn how it works

- [Sparse graph topology](concepts/topology.md) explains COO, CSR, transpose,
  lowering, and the push-pull tradeoff.
- [Message passing](concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Sparse aggregation feasibility](research/sparse-aggregation.md) records the
  revision-bound implementation and scaling evidence.
- [Mean GraphSAGE experiment](research/mean-sage.md) records the exact learning
  witness and its limits.

Concept pages describe ideas that should survive an implementation change.
Research records bind claims to exact revisions and measurements. Source and
tests own current behavior.
