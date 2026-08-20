# GTFS semantic boundary

This record asks whether a fixed-stop GTFS Schedule plus one GTFS Realtime
snapshot fits TinyMesh's existing sparse primitives. It does. Stage 1 needs no
public GTFS API, dynamic graph, ontology runtime, dataframe, or protobuf
dependency.

## Boundary

The official [GTFS Schedule reference](https://gtfs.org/documentation/schedule/reference/)
and [GTFS Realtime reference](https://gtfs.org/documentation/realtime/reference/)
own transport semantics. The pinned Google sample and the declared realtime
fixture are executable witnesses, not a replacement specification.

```text
Schedule ZIP                 Realtime full snapshot
    |                                 |
    v                                 v
validated immutable host facts   normalized host facts
    |                                 |
    +------ route segments             +------ transition eligibility
                  |                                  |
                  v                                  v
          Graph(N, source[E], target[E])   delay/mask/vehicles [N,1]
```

Host facts remain the source of truth. The graph, node rows, edge positions,
masks, tensors, and graph version are deterministic derived views.
Realtime facts carry the exact Schedule revision and checksum against which
their identities were resolved; lowering rejects a different manifest even
when its external source name is unchanged.

## Representation

Each graph node is one fixed stop. Each graph edge is one
`(route_id, from_stop_id, to_stop_id)` relation. Keeping the route in the edge
identity prevents equal stop patterns on different routes from collapsing.
Repeated trip-call occurrences may share that edge because the edge map retains
every contributing `(trip_id, from_sequence, to_sequence)` occurrence.

The graph carries only sparse computation. Stop and route identifiers,
service-day time, trip-instance identity, provenance, predictions,
cancellations, transition findings, and occurrence evidence stay on the host
boundary. This is also where a future mapping to the
[Common Core Ontologies](https://github.com/CommonCoreOntology/CommonCoreOntologies)
belongs; an ontology term is not a tensor coordinate or a reason to widen
`Graph`.

At one observation time, Stage 1 derives three node-aligned fields:

| Field | Shape | Type | Meaning |
| --- | --- | --- | --- |
| delay | `[N,1]` | float32 | current trip delay at the declared vehicle progression stop |
| observed | `[N,1]` | bool | whether delay is an eligible observation |
| vehicle count | `[N,1]` | int32 | eligible vehicle present at that stop |

Numeric zero never means observed. Missing, stale, canceled, contradictory, or
ineligible state leaves the mask false. Future stop predictions remain host
facts and do not mutate planned topology or current node state.

## Evidence

Revision `e97af2f` was the dependency baseline for the integrated experiment.
The pinned sample lowers to 9 stop nodes and 15 route-bearing segment edges.
The declared realtime fixture places one vehicle and one observed 120-second
delay at `NANAA`; all other delay rows remain zero with a false mask.

The retained edge map reconstructs all 15 relations and every contributing
trip-call occurrence. Independent host loops reproduce the node, edge, delay,
mask, and vehicle-count outputs. Tests also cover source reordering, bijective
stop relabeling, parallel routes, repeated occurrences, missing state, stale
trip state, stale vehicle state, cancellation, and a changed call occurrence.
A call-occurrence change produces a new graph version even when COO topology is
unchanged, so cached planned semantics cannot be silently reused.

The stored carriers are `O(N + E)`: two COO index tuples and an edge map of
length `E`, plus three `[N,1]` tensors. No `[N,N]` or `[N,E]` carrier is created.
The exact witness runs on CPU and Metal through ordinary tinygrad tensors.

```console
uv run --locked python -m experiments.run gtfs_snapshot DEV=CPU
uv run --locked python -m experiments.run gtfs_snapshot DEV=METAL
```

## Decision

The existing `Graph` and ordinary tinygrad tensors are sufficient for this
boundary. Adding semantic fields to `Graph`, creating a GTFS-specific core
wrapper, or promoting experiment types into `src/tinymesh/` would duplicate
host ownership without improving sparse computation.

Reopen the decision only when evidence requires at least one of:

- topology that changes within a modeled sequence;
- flexible locations or geometry participating directly in computation;
- multiple vehicles requiring a defined aggregation contract;
- heterogeneous node or relation types with distinct operators;
- durable temporal/event memory rather than one bounded snapshot;
- a stable public loader needed by more than research experiments.

Until then, new Schedule and Realtime semantics should extend the validated
host boundary and derive the same small sparse carriers.

## Limits

The sample is tiny, frequency-based, fixed-stop, and contains one vehicle. It
does not establish a production ingestion API, live-feed reliability,
multi-vehicle aggregation, learned anomaly detection, or model quality. CPU and
Metal execution prove only that this exact lowering and tensor realization work
on both backends.

The [GTFS Realtime best practices](https://gtfs.org/documentation/realtime/realtime-best-practices/)
motivate the explicit freshness policy used by the transition witness. Policy
values remain evaluator inputs rather than hidden defaults in the projection.
