# MBTA departure-event mesh

Status: Stage 1 carrier retained; longer replay acquisition remains unstarted.

## Decision

Retain the ragged departure-event mesh as the source carrier. A 30- or
60-second clock preserves the 575 target identities in this slice, but spends
most of its cells on absence. A five-minute clock is compact but merges 90
events and therefore requires an aggregation policy. Clocks remain derived,
matched controls rather than source truth.

```text
source rows + Schedule
          |
          v
  physical departures       597 records
      /          \
     v            v
  `run` 534    `headway` 575
      \          /
       v        v
   causal prefix(cutoff)

clock projection (derived only)
  30 s  -> lossless, 89.1% empty
  60 s  -> lossless, 78.2% empty
 300 s  -> 90 identities merged
```

This is a representation decision, not a forecast, topology-value, MBTA
service-quality, online-causality, or public API claim.

## Contract

One event key is:

```text
(service_date, vehicle_id, parent_station,
 direction_id, departure_timestamp)
```

The departure timestamp is the following adjacent Schedule call's observed
Vehicle Position movement timestamp. Source trip instance, stop identity,
sequence, following call, Schedule departure, and source headway remain aliases
on the physical event. Thus every retained event maps back to its exact source
row and both applicable Schedule calls.

`run(a, b)` connects consecutive observed station departures in one trip and
vehicle trace. `headway(a, b)` connects consecutive physical departures in one
parent-station, trunk-route, and direction lane. Both relations name their
endpoints and elapsed seconds and require `time(a) < time(b)`.

`prefix(c)` is the event-induced subgraph whose timestamps are strictly below
`c`. It cannot contain an edge incident to an excluded target because relation
retention is derived only from the retained event-key set.

## Evidence

The clean, locked run was:

```console
uv run --locked python -m experiments.run gtfs_event_mesh
```

| Revision | Result |
| --- | --- |
| `98c77969b46e394427075bd1632298740fd5df8b` | retained event carrier |

The envelope binds the structural comparison to PyG
`5c6461b2305ad068a6d61165b3c55852a11aaa41`, PyG Temporal
`fe555bc30ee197755c4b58a89407033a5f383415`, and TSL
`aa5f313e000d192bdec270748b8d01df5912e58e`. No framework is imported at
runtime.

### Exact mesh

| Measure | Result |
| --- | ---: |
| source / represented rows | 663 / 597 |
| physical departures / aliases / duplicate aliases | 597 / 597 / 0 |
| `run` / `headway` relations | 534 / 575 |
| source / exact / derived-only headways | 608 / 575 / 0 |
| boundary-only source headways | 33 |
| maximum in-degree / out-degree | 2 / 2 |
| event + relation records | 1,706 |
| unconstructed dense event-pair cells | 356,409 |

All 575 internally derivable source headways reproduce exactly. The other 33
labels need an event beyond the retained cut and remain boundary evidence; none
is imputed. A midpoint witness retains 297 events and 536 relations, excludes
300 events, and identifies 30 crossing relations that the prefix omits.

The lowering groups only trip traces and station-direction lanes, sorts within
those groups, and emits adjacent relations. It never constructs station-pair,
event-pair, or full station-time arrays.

### Clock comparison

The comparison uses the same 575 exact target events and 22 lanes over the
declared two-hour interval. One scalar cell can retain at most one event
identity without a sidecar.

| Clock | Cells | Empty | Occupied | Colliding | Merged identities | Max/cell | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30 s | 5,280 | 4,705 | 575 | 0 | 0 | 1 | lossless but 89.1% empty |
| 60 s | 2,640 | 2,065 | 575 | 0 | 0 | 1 | lossless but 78.2% empty |
| 300 s | 528 | 43 | 485 | 90 | 90 | 2 | reject as source carrier |

The small clocks do not lose identity in this witness, but neither is simpler
source truth: each introduces an arbitrary cadence, missing-cell semantics,
and several times more storage than the target events. The five-minute clock
cannot be lossless without a ragged sidecar, which recreates the event carrier.

## Structural comparison

| TinyMesh fact | Pinned framework correspondence | Where correspondence breaks |
| --- | --- | --- |
| one typed relation | PyG `TemporalData(src, dst, t, msg)` can encode source key, target key, target time, and relation payload on one event axis | PyG does not own physical transit identity, aliases, provenance, relation vocabulary, or cutoff semantics |
| ordered relation rows | PyG `TemporalDataLoader` batches consecutive rows and preserves order | batching is not a causal history/target split; its negative destination sampling is irrelevant here |
| one clock slice | PyG Temporal `StaticGraphTemporalSignal` iterates fixed-topology feature and target snapshots | it explicitly assumes constant-time snapshots and cannot decide collision or absence semantics |
| lane-time target and mask | TSL `SpatioTemporalDataset` exposes target, index, mask, connectivity, covariates, window, delay, and horizon | those fields presuppose a chosen clock; the container cannot establish event identity, aggregation, or fact availability |

These are useful contract comparators, not architectural owners. The Stage 1
host records stay experiment-local, and `src/tinymesh/`, dependencies, and
exports remain unchanged.

## Malformed witnesses

Tests prove deterministic lowering and exact Schedule/source round trips. A
duplicate trip alias collapses only when its physical key agrees. Conflicting
alias labels, unresolved or contradictory Schedule identity, non-forward time,
self edges, duplicate relations, contradictory relation kinds, and unknown
endpoints fail with the relevant identity. Checksum or artifact drift fails in
the retained loader before lowering. Prefix tests exclude the target and every
future incident relation.

## Stage 2 consequence

Acquire and retain event facts, not pre-binned snapshots. Stage 2 should first
estimate a capped 28-day, all-rapid-transit extraction, then retain the smallest
population that covers ordinary and disrupted service without weakening
observational lineage. Its manifest must preserve:

- immutable raw checksums, license, source revision, active Schedule, route,
  service date, trip instance, vehicle, stop sequence, parent station,
  direction, trunk route, and movement timestamp;
- exact Schedule-call identity as a separate mask: absence may disable
  plan-derived features or baselines but cannot erase an observed movement;
- source headway only as an audit label, plus explicit generation and ingestion
  clocks when the source actually provides them;
- enough leading and trailing context to derive departures and headways at the
  evaluation boundary without imputation;
- duplicate physical aliases and conflicts as measured outcomes; and
- event counts, lane rates, gaps, boundary loss, regime coverage, and byte cost
  before Stage 3 chooses windows, splits, clocks, or models.

If availability clocks cannot be recovered, the eventual claim remains
retrospective event-time forecasting. The 30- and 60-second projections may be
reintroduced in Stage 3 as matched controls, after the causal task and
aggregation policy are frozen.

## Limits

This witness is one line, one weekday, and two hours. Empty-cell and collision
rates may change materially by line, service regime, and duration. The rare
duplicate-alias case is synthetic here, though the retained seven-day source
audit found one real duplicate. The experiment proves carrier mechanics and
source-label equivalence only; it does not establish forecast sufficiency.
