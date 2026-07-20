# tinymesh

**Graphs through time, in tinygrad.**

tinymesh is an experimental library for learning over sparse spatiotemporal
graphs using [tinygrad](https://github.com/tinygrad/tinygrad).

A mesh is a sparse graph topology together with time-indexed node and edge
state. The initial focus is deliberately narrow: fixed graphs, temporal
windows, message passing, and recurrent forecasting.

## Principles

- Sparse by construction: computation scales with edges, not node pairs.
- Tensor-first: graph and temporal state have small, explicit contracts.
- Tinygrad-native: no compatibility layer over another ML framework.
- Composable: message, aggregate, update, and temporal evolution remain separate.
- Minimal: one clear primitive before multiple architectures or abstractions.

## Scope

The first milestone is:

1. A sparse graph record.
2. A temporal graph record.
3. An edge-based message-passing primitive.
4. One recurrent spatiotemporal forecasting model.
5. One reproducible synthetic example.

tinymesh is not an application, GIS SDK, data platform, trainer framework, or
model zoo.

## Status

Early research. There is no stable API or installable release yet.

The `tinygrad` submodule is a pinned source reference for implementation study.
It is not vendored application code.
