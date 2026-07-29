# Montevideo spatial-temporal data

The Montevideo bus dataset is the first real source with topology, metric
position, scalar edge facts, and an ordered node signal. This stage starts at
the source boundary. Tensor lowering follows only after the host values are
validated and aligned.

## Pinned source

Tinymesh reads
[PyG Temporal `fe555bc`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/dataset/montevideo_bus.json)
with SHA-256
`37d9c6286d474077b5c05173c1570c4da42c387013116daa8862c7a6cab86a75`.
The JSON payload is `3,131,884` bytes. The reader accepts at most 4 MiB and
checks that limit before JSON parsing.

An explicit path supports offline runs and fixtures. The default path uses a
10-second standard-library request, reads at most 4 MiB, and verifies the exact
checksum above. Python's standard library owns acquisition, JSON parsing, and
source validation. No geo or dataframe package enters the runtime.

## Source contract

```text
ordered nodes
  bus_stop            stable source identity
  lon, lat            two finite numeric position values
  X.y                 ordered feature observations
  y                   ordered target observations

ordered links
  source, target      directed bus_stop identities
  weight              positive finite road distance
```

The parser resolves link identities to node rows while preserving both source
orders. It rejects duplicate node identities, duplicate directed links, missing
endpoints, non-finite geometry, non-positive distance, unequal time axes, and
oversized input. It returns plain immutable Python values; it creates no tensor,
normalization statistic, or dense topology.

The source calls its projected position fields `lon` and `lat`. Tinymesh
preserves the numeric values without interpreting or transforming them at this
boundary. Coordinate-frame ownership belongs to the aligned dataset lowering.

## Full-source witness

```console
uv run --locked python -m experiments.montevideo_source [path]
```

At the pinned revision:

```json
{
  "nodes": 675,
  "edges": 690,
  "steps": 744,
  "duplicate_edges": 0,
  "self_loops": 0,
  "position_dimensions": 2,
  "minimum_road_distance": 23.8,
  "maximum_road_distance": 1991.3
}
```

This establishes source structure, not model quality or framework parity.
