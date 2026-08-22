# MBTA event population

Status: Stage 2 stopped; Stage 3 is blocked on recoverable Schedule identity.

## Decision

Do not specify or train a headway forecast from this population. The bounded
28-day acquisition has ample rows, dates, routes, and source headways, but
21.8% of rows cannot round-trip to the active Schedule identity exposed by the
public daily source. Filtering those rows would disproportionately erase added
and disrupted service.

```text
28 dates × 8 routes
      1,050,259 rows
            |
      +-----+------+
      |            |
      v            v
  821,513       228,746
  resolved      unresolved
                   |
          +--------+--------+
          |        |        |
       221,220   4,363    3,163
        ADDED   NONREV     other
                   |
                   v
     stop: insufficient Schedule identity
                   |
                   v
         do not open Stage 3
```

This is a provenance failure, not a claim that the operational observations
are inaccurate or that headway is unforecastable.

## Bounded acquisition

The pure planner selected every complete service date from 2026-07-24 through
2026-08-20 in the official [LAMP daily-file
index](https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/index.csv).
The index snapshot was observed under cutoff `2026-08-22T20:24:04+00:00` and
has SHA-256
`e28e9c404518fcaea470e0032411e91a8c6496bfd6bba5b8f2fafacfcd6910d4`.

| Source | Files | Bytes |
| --- | ---: | ---: |
| daily rapid-transit performance | 28 | 34,175,215 |
| shared 2026 Schedule and service calendars | 6 | 76,434,973 |
| total | 34 | 110,610,188 (105.49 MiB) |
| hard cap | — | 134,217,728 (128 MiB) |

The initial estimate omitted `calendar.parquet` and
`calendar_dates.parquet`. They add 1,072,269 bytes and are necessary to decide
which service IDs operated on each date. The corrected population remains
18.02% below the cap.

Acquisition downloads into a sibling staging directory, bounds each response
by its declared size and a 30-second socket timeout, validates already-known
Schedule checksums, computes SHA-256 for every daily source, and publishes the
directory only after all 34 sources seal. Raw parquet and operational rows
remain outside Git. The repository retains only a 16 KiB source manifest and
an 88 KiB date-route audit.

## Source contract

The [official public catalog](https://performancedata.mbta.com/) defines daily
performance partitions and the annual Schedule archive. The [LAMP data
dictionary](https://github.com/mbta/lamp/blob/main/Data_Dictionary.md) states
that daily `trip_id` comes from GTFS Realtime, while movement timestamps come
from Vehicle Positions and scheduled fields come from GTFS.

At pinned LAMP revision
[`e266440d`](https://github.com/mbta/lamp/tree/e266440db994ed33eede5e44a137b205e4a1e8dd),
the exporter joins Schedule fields through the internal
`VehicleTrips.static_trip_id_guess` and `static_version_key`. Its backup matcher
maps added real-time trips to a nearest scheduled trip by route, direction, and
start time. The public daily parquet includes neither the resulting static trip
identifier nor the version key.

That omission matters because an `ADDED-*` real-time trip ID is not itself a
Schedule trip ID. The daily row retains derived scheduled values but not the
identity needed to independently recover the exact applicable calls. Treating
those values as an unnamed Schedule would weaken the reversible identity
contract established in Stage 1.

The larger public `LAMP_ALL_RT_fields.parquet` does expose
`static_trip_id_guess` and `static_version_key`, but its observed object size was
4,412,388,321 bytes (4.11 GiB), over 32 times the entire Stage 2 cap. It is one
mutable monolithic object rather than immutable daily partitions, so Stage 2
did not acquire or silently range-project it.

## Population audit

| Measure | Result |
| --- | ---: |
| service dates / routes / active Schedule versions | 28 / 8 / 12 |
| source rows / trip instances | 1,050,259 / 98,767 |
| source headways | 941,984 |
| rows missing movement timestamps | 55,164 (5.25%) |
| exact active-Schedule rows | 821,513 (78.22%) |
| unresolved Schedule rows | 228,746 (21.78%) |
| unresolved `ADDED-*` / `NONREV-*` / other | 221,220 / 4,363 / 3,163 |

The problem is not diffuse harmless noise. Examples include every one of the
5,112 Green-D rows on 2026-08-12 and all 4,276 Green-D rows on 2026-08-14.
Removing unresolved rows would remove whole route-days and preferentially
discard atypical operations. A model trained afterward could appear clean only
because the acquisition deleted the regimes it was meant to test.

The daily source exposes event time but no per-event generation, ingestion, or
as-of clock. `index.csv.last_modified` is publication metadata for a completed
file, not fact availability. Even with Schedule identity repaired, the current
population could support only retrospective event-time forecasting.

## Reproduction

The acquisition commands require explicit temporary or external directories:

```console
uv run --locked experiments/tools/mbta_population.py plan --observed-at 2026-08-22T20:24:04+00:00 --output /tmp/mbta-population-plan.json
uv run --locked experiments/tools/mbta_population.py acquire --plan /tmp/mbta-population-plan.json --source-dir /tmp/mbta-population-source
uv run --locked --with duckdb==1.4.1 experiments/tools/mbta_population.py record --source-dir /tmp/mbta-population-source --output-dir experiments/fixtures/mbta_population
uv run --locked python -m experiments.run mbta_population
```

The clean decision run is bound to revision
`f31c48a6e436dd839c67264ba4b243f905ba9930`. It reads no study gitlink and
records an empty reference set.

## What would reopen Stage 3

Any replacement must preserve the event carrier and satisfy one of these
without exceeding an explicit cap:

- immutable date-partitioned rows containing the matched static trip ID and
  Schedule version;
- an official immutable mapping from real-time trip aliases to exact Schedule
  identities; or
- a fully reproducible mapping whose output is unique and agrees with every
  exported Schedule fact, including added and disrupted service.

Until then, retain the Stage 1 carrier and this negative population record, but
do not add a task, model, public loader, clock default, or partial “clean”
dataset. Reducing dates or routes does not repair the ownership hole.

## Limits

The audit proves failure for one explicit four-week 2026 population and the
pinned public source contract. LAMP could publish a smaller identity-bearing
partition later. The result does not reject other transit operators, direct
GTFS-Realtime archives with feed timestamps, or a future MBTA export with the
missing identity and availability fields.
