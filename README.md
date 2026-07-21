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

Sparse aggregation is not yet admitted: the native Tinygrad candidate is
correct and differentiable, but its work scales with node-edge pairs. The
[feasibility record](docs/sparse-aggregation.md) contains the reproducible gate
and exact evidence.

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
```

Initialize the optional study references with:

```console
git submodule update --init
```
