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

Early research. There is no stable API or installable release yet.

The `tinygrad` submodule is a pinned source reference for implementation study.
It is not vendored application code.
