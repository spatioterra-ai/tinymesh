# MBTA clock boundary

Status: Stage 0 closed the snapshot branch.

## Decision

`stop:no_identity_preserving_clock`

No 30-, 60-, or 300-second clock preserves the frozen retrospective event
task. The finest candidate already merges 1,459 exact predecessor-target pairs
and contains 1,461 bins with strictly ordered departures. Coarser clocks lose
more identity. Stage 1's snapshot DiffusionGRU branch is therefore closed; the
event carrier remains canonical.

```text
947,489 physical departures ---> 940,551 strict targets
             |
             +--  30 s --> 1,459 merged targets --+
             +--  60 s --> 3,823 merged targets --+--> stop snapshot branch
             `-- 300 s --> 120,752 merged targets -+
             |
             `----------------------------------------> retain event-time mesh
```

A regular clock is not merely a lossy storage choice here. One snapshot update
cannot represent two strictly ordered departures in the same operational lane
and bin. Averaging or selecting one departure would change the task rather than
provide a matched control.

## Frozen boundary

The audit rehydrated only the 110,610,188 bytes named by the sealed population
manifest and verified every retained source fact before projection. Rebuilding
the task had to reproduce its protocol byte-for-byte; the topology protocol had
to name that exact task digest and 940,551 targets. This also freezes the split,
route counts, exclusions, and retained Schedule/run-ambiguity masks.

Events are grouped by `(service_date, parent_station, trunk_route_id,
direction_id)` in half-open UTC bins `[start, end)`. Empty work spans only the
inclusive range from a lane-day's first occupied bin through its last. Equal
timestamps remain separate physical events and are reported independently; no
order is invented for them.

## Full-population audit

| Clock | Cells | Occupied | Empty | Empty rate | Causal collision bins / events | Merged targets | Maximum occupancy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 s | 14,333,994 | 945,817 | 13,388,177 | 93.40% | 1,461 / 2,932 | 1,459 | 3 |
| 60 s | 7,170,046 | 943,452 | 6,226,594 | 86.84% | 3,811 / 7,648 | 3,823 | 3 |
| 300 s | 1,438,867 | 826,474 | 612,393 | 42.56% | 102,906 / 223,764 | 120,752 | 6 |

The 206 equal-time sets containing 413 events are unchanged across candidates
and are excluded from causal-collision counts. Collision bins and merged
targets need not match one-for-one: a bin may affect multiple targets, while a
collision at a lane-day boundary may not correspond to an eligible target.

Merged targets occur in every split:

| Clock | Train | Validation | Test |
| ---: | ---: | ---: | ---: |
| 30 s | 1,087 | 94 | 278 |
| 60 s | 2,915 | 187 | 721 |
| 300 s | 82,941 | 12,973 | 24,838 |

They are also not confined to one route:

| Route | 30 s | 60 s | 300 s |
| --- | ---: | ---: | ---: |
| Blue | 2 | 2 | 8,897 |
| Green-B | 411 | 1,083 | 15,299 |
| Green-C | 334 | 913 | 16,023 |
| Green-D | 376 | 817 | 20,491 |
| Green-E | 166 | 590 | 20,852 |
| Mattapan | 19 | 41 | 781 |
| Orange | 10 | 22 | 22,916 |
| Red | 141 | 355 | 15,493 |

The strongest 30-second collision concentrations are Green lane-days at
`place-kencl` / direction 1 and `place-pktrm` / direction 0: four named
lane-days each contain 22 collision bins and 44 participating events. The
largest 30-second work extreme is `place-clmnl` / Green / direction 1 on
2026-07-27, with 34 occupied cells inside a 2,712-cell active span. These
extremes reinforce the global result but do not drive it; the admission rule is
exact over every target.

## Reproduction

The retained audit is generated only from an explicitly rehydrated source
directory:

```console
uv run --locked -m experiments.tools.mbta_population acquire --plan experiments/fixtures/mbta_population/manifest.json --source-dir /tmp/mbta-population-source
uv run --locked --with duckdb==1.4.1 -m experiments.tools.mbta_clock --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --task-protocol experiments/fixtures/mbta_headway_task/protocol.json --topology-protocol experiments/fixtures/mbta_topology/protocol.json --output /tmp/mbta-clock-audit.json
uv run --locked python -m experiments.run mbta_clock
```

The clean decision run is bound to tinymesh revision
`f95a77971ddb800981fbfdcfd636fa8c6681f766` and pinned PyTorch Geometric
Temporal revision `fe555bc30ee197755c4b58a89407033a5f383415`. The reference
iterator establishes the snapshot-sequence interface only; it does not supply
an aggregation or missingness policy that could repair event identity.

## Scope and consequence

This result rejects these three regular clocks as matched carriers for this
frozen 28-day task. It does not reject irregular event-time recurrence, clocks
for a newly specified aggregate target, or clocks below 30 seconds. A finer
clock would increase the already 15.13 cells per physical event at 30 seconds
and was outside the fixed candidates.

No runtime dependency, dataset adapter, public export, tensor model, or
`src/tinymesh/` code is introduced. The next stage must specify an event-time
memory control directly and preserve the frozen task and topology boundary.
