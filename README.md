# tinymesh

**Sparse structure through space and time, in tinygrad.**

tinymesh is an experimental library for learning over graphs, geospatial
structures, and geometric meshes using
[tinygrad](https://github.com/tinygrad/tinygrad).

A mesh combines sparse topology, optional geometry, attached tensor fields, and
time. A graph is the simplest mesh. Coordinates embed it in geographic or
physical space; faces and volumes extend it into 2D and 3D; time-varying fields
or geometry make it 4D.

The initial implementation remains deliberately narrow: fixed graph topology,
optional coordinates, temporal fields, message passing, and recurrent
forecasting. Higher-dimensional cells and changing topology follow only when
the core provides a natural extension.

## Principles

- Sparse by construction: computation scales with edges, not node pairs.
- Tensor-first: topology, geometry, fields, and time have explicit contracts.
- Geometry is optional: topology remains useful without an embedding.
- Geo-aware: coordinates and spatial-reference metadata are first-class.
- Tinygrad-native: no compatibility layer over another ML framework.
- Composable: message, aggregate, update, and temporal evolution remain separate.
- Minimal: one clear primitive before multiple architectures or abstractions.

## Scope

The first milestone is:

1. A sparse vertex-and-edge topology.
2. Optional coordinates and attached tensor fields.
3. Fixed-topology, time-indexed fields.
4. An edge-based message-passing primitive.
5. One recurrent spatiotemporal model and reproducible synthetic example.

Geospatial learning is core scope. General-purpose GIS storage, file parsing,
reprojection, and cartographic rendering enter through adapters. tinymesh is not
an application, data platform, trainer framework, or model zoo.

## Status

Early research. The package is installable from source, but there is no stable
API or published release yet.

Tinygrad's public gather-and-scatter composition still scales with node-edge
pairs. A Tinymesh-owned destination-CSR custom kernel now passes the recorded
forward, gradient, and edge-linear structure gates on CPU and Metal; its
transpose CSR supplies the backward pass without atomics. It remains an
experiment because `Tensor.custom_kernel` is alpha and the required dynamic
loop is new in the pinned Tinygrad revision; Tinygrad's default kernel option
search also still fails for this loop, so the candidate disables it. The
[feasibility record](docs/sparse-aggregation.md) contains the exact boundary and
evidence.

The submodules are pinned, reference-only source for implementation study:

- `tinygrad` matches the exact runtime dependency and lockfile revision;
- PyTorch Geometric 2.8.0 and PyTorch Geometric Temporal `fe555bc` provide
  comparative graph and temporal implementations.

They are not vendored runtime code. PyTorch itself is not a submodule: its
repository is much larger, while exact links to the indexed, scatter, segment,
and sparse-matrix operators provide the useful study boundary.

Tinygrad development follows upstream `master`. The runtime dependency,
lockfile, and reference submodule pin the same commit so each experiment remains
reproducible; all three advance together at the start of a new stage. The
comparative references move only with an intentional, revision-bound study.

## Development

Requires [uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python experiments/sparse_aggregation.py
uv run python experiments/csr_aggregation.py
```

Initialize the optional study references with:

```console
git submodule update --init
```

Contributions follow
[CONTRIBUTING.md](CONTRIBUTING.md).
Repository prose, issues, pull requests, and reviews follow
[voice.md](voice.md).
