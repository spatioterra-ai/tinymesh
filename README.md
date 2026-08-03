<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/assets/logo_mesh_light.svg">
  <img alt="tinymesh" src="/docs/assets/logo_mesh_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinymesh

**Sparse structure through space and time, in tinygrad.**

tinymesh is an experimental library for learning over graphs and meshes with
[tinygrad](https://github.com/tinygrad/tinygrad). A graph is the smallest mesh:
sparse topology connects tensor fields. Geometry and time extend that core
without replacing it.

tinymesh has one runtime dependency, tinygrad. It is tinygrad-native, not a
compatibility layer over another machine-learning framework.

[Documentation](https://spatioterra-ai.github.io/tinymesh/) |
[Quick start](https://spatioterra-ai.github.io/tinymesh/quickstart/) |
[API](https://spatioterra-ai.github.io/tinymesh/api/) |
[Contributing](CONTRIBUTING.md)

[![Tests](https://github.com/spatioterra-ai/tinymesh/actions/workflows/tests.yml/badge.svg)](https://github.com/spatioterra-ai/tinymesh/actions/workflows/tests.yml)

## What works

```text
ordered COO edges + node tensors
               |
          lower once
               v
     CSR(A) + CSR(A.T) + edge maps
               |
       +-------+--------+
       |                |
       v                v
  sparse fields    sparse aggregation
  endpoints        sum / mean / edge sum / weighted sum
  target softmax          |
       +------------------+
               |
               v
     spatial + temporal components
```

- `Graph` owns deterministic directed topology, sparse node and edge sums,
  endpoint projection, target softmax, in-degree, and scalar edge identity.
- Forward and first-order backward store `O(N + E)` topology and perform
  `O((N + E)H)` work without dense adjacency or node-edge carriers.
- Leading axes share one graph, so `Graph.sum` accepts `[..., N, H]`.
- `tinymesh.nn` composes direct tinygrad-style node- and edge-aware graph
  convolution, attention, recurrence, period attention, and directed diffusion.
- `StaticGraphTemporalSignal` and the pinned Chickenpox, Montevideo, and
  METR-LA loaders keep graph, node, time, feature, target, and edge axes aligned.
- CPU and Metal follow the same checked contracts.

The [API reference](docs/api.md) is generated from the source. The current
[research ledger](docs/research/index.md) separates what executes from what the
evidence supports:

```text
controlled transport   correct topology matters and transfers across graph size
real forecasts         METR-LA topology signal; incumbent local model still wins
implementation         fixed-topology first-order core; alpha custom kernel
```

This is experimental 0.x code, not a stability promise. The private CSR backend
uses alpha `Tensor.custom_kernel`; tinygrad's default kernel optimization does
not yet accept its data-dependent loop.

## Try it

Install the locked tinygrad revision with [uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
```

```python
from tinygrad import Device, Tensor
from tinymesh import Graph

graph = Graph(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device=Device.DEFAULT).realize()

print(graph.sum(state).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Layers are ordinary callable objects:

```python
from tinymesh.nn import SAGEConv

layer = SAGEConv(in_features=1, out_features=2)
print(layer(state, graph).shape)
# (4, 2)
```

List the revision-bound experiments:

```console
uv run --locked python -m experiments.run --list
```

Successful runs write ignored local envelopes containing the tinymesh revision,
all reference pins, explicit settings, execution bounds, and the JSON
observation. See [Experiments](docs/experiments.md).

## Learn

- [Quick start](docs/quickstart.md) follows one value through sparse execution,
  gradients, layers, and time.
- [API](docs/api.md) is the source-generated public reference.
- [Concepts](docs/concepts/topology.md) explain topology,
  [message passing](docs/concepts/message-passing.md), and
  [time](docs/concepts/time.md).
- [Research](docs/research/index.md) owns the current decision ledger and links
  every revision-bound record.
- [Papers](docs/papers.md) pins publication metadata and exact source revisions
  without committing downloaded PDFs or TeX.

## Repository

```text
src/tinymesh/    public runtime: sparse math, direct layers, data boundaries
experiments/     revision-bound policy, controls, training, measurements
tests/           executable contracts
docs/concepts/   durable theory
docs/research/   exact evidence and current decisions
papers/          tracked citations and exact-source fetch, ignored cache
submodules/      pinned read-only references
```

The runtime never imports experiments or submodules. Research may use public
runtime objects; results change the API only after the documented graduation
gate passes.

## Development

```console
uv sync --locked
uv run --locked python -m unittest discover -s tests -p 'test_*.py'
uv run --locked --only-group lint ruff check .
uv run --locked --only-group lint mypy
uv build
```

Build or preview the docs with the locked docs environment:

```console
uv run --locked --only-group docs zensical build --clean --strict
uv run --locked --only-group docs zensical serve
```

Pinned submodules are optional, reference-only source:

```console
git submodule update --init
```

Their roles and exclusions live in
[Reference projects](docs/reference-projects.md). See
[CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
