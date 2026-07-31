<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/logo_mesh_light.svg">
  <img alt="tinymesh" src="assets/logo_mesh_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinymesh

tinymesh is a tinygrad-native library for learning over sparse structures
through space and time.

A graph is the smallest mesh. Nodes carry tensor fields and edges say which
nodes may interact. Coordinates, higher-dimensional cells, and time can extend
that structure without replacing the sparse core.

## Start here

| Goal | Read |
| --- | --- |
| Run one graph end to end | [Quick start](quickstart.md) |
| Look up a class or function | [API](api.md) |
| Understand core theory | [Topology](concepts/topology.md), [message passing](concepts/message-passing.md), [time](concepts/time.md) |
| See what the evidence supports | [Research](research/index.md) |
| Reproduce an observation | [Experiments](experiments.md) |

## Public boundary

| Import | Owns |
| --- | --- |
| `tinymesh` | `Graph`, `StaticGraphTemporalSignal` |
| `tinymesh.nn` | reusable equations and parameters |
| `tinymesh.datasets` | pinned source validation and tensor lowering |
| `experiments` | non-runtime data policy, training, controls, and claims |

Neural-network components are direct objects with ordinary tinygrad `Tensor`
attributes and `__call__`. There is no factory, registry, trainer, or PyTorch
compatibility surface.

## The stack

```text
edge facts             source -> target, optional COO scalar
    |
    v
topology lowering      COO -> CSR(A) + CSR(A.T) + edge maps
    |
    v
sparse operations      endpoint fields, target softmax, sum / mean
    |
    v
spatial composition    position -> displacement -> distance -> weight
    |
    v
temporal alignment     Graph + x[T,N,F] + y[T,N,Y]
    |
    v
model composition      graph convolution, attention, recurrence, diffusion
```

[tinygrad](https://github.com/tinygrad/tinygrad) owns tensors, autograd,
compilation, and device execution. tinymesh owns sparse topology, mesh
semantics, and the model compositions that need them.

Components compose in ordinary Python:

```text
spatial state -----+
spectral state ----+--> same-shaped tensors --> sum / concat / attention --> task head
temporal state ----+
future multiscale -+
```

Task heads, training, and combination policy stay outside the library until a
reusable invariant needs an owner.

## Current boundary

The fixed-topology core supports unit and scalar-weighted aggregation,
first-order gradients, target-normalized attention, fixed-graph temporal
signals, and direct graph-temporal components on CPU and Metal. It stores
`O(N + E)` topology and never constructs dense adjacency.

The private CSR backend uses alpha tinygrad `Tensor.custom_kernel` and disables
default kernel optimization for its data-dependent loop. Batching different
graphs, changing topology, vector edge messages, higher-order gradients,
general coordinate-reference machinery, geodesy, and higher-dimensional cells
remain outside the current contract.

Concept pages hold durable theory. Research records bind claims to exact
revisions and measurements. Source and tests own current behavior.
