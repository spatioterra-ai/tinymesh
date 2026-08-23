# MBTA event population

Status: Stage 2 passed for retrospective event-time forecasting.

## Decision

Advance to Stage 3 without filtering the source population. The bounded 28-day
acquisition has ample dates, routes, movement rows, and source headways for a
retrospective event-time task. Exact active-Schedule identity is available for
78.2% of rows; Stage 3 must mask Schedule-dependent features and baselines on
the remainder rather than erase added and disrupted service.

```text
28 dates × 8 routes
      1,050,259 rows
            |
            v
  observable trip ordering
     |                |
     v                v
947,489 events   384 ambiguous rows
     |
     v
940,776 exact movement-headway labels
     |
     +---------------------------+
     |                           |
     v                           v
retrospective Stage 3      Schedule identity mask
                            821,513 / 228,746
```

Physical-event identity and Schedule-call identity are different facts. The
former supports the event carrier and 940,776 exact eligible headway labels.
The latter identifies the exact plan used for optional plan-derived features
and baselines. Missing Schedule identity limits those comparisons; it does not
invalidate an observed movement.

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
a 208 KiB date-route audit.

```text
Plan[optional expected checksums]
  | acquire or seal
  v
Manifest[required observed checksums]
  | verify bytes
  v
Population tables
```

The Python lifecycle makes these states distinct even though both artifacts
retain the flat schema-v1 JSON layout. A plan cannot be opened as a population;
only a fully checksummed manifest reaches verification and table construction.

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
identity needed to independently recover the exact applicable calls. Stage 3
therefore cannot present those rows as exact Schedule round trips. It can still
use their observed movement events and movement-derived headways.

The larger public `LAMP_ALL_RT_fields.parquet` does expose
`static_trip_id_guess` and `static_version_key`, but its observed object size was
4,412,388,321 bytes (4.11 GiB), over 32 times the entire Stage 2 cap. It is one
mutable monolithic object rather than immutable daily partitions, so Stage 2
did not acquire or silently range-project it.

## Event contract

The public trip identity is `(service_date, route_id, trip_id)`, matching the
unique trip key in pinned LAMP. Within each trip, LAMP orders rows by a
coalesced movement/stop event timestamp and defines a station departure as the
following row's Vehicle Position movement timestamp. The audit applies the
closest publicly reproducible rule and lowers the result to the Stage 1
physical key:

```text
(service_date, vehicle_id, parent_station,
 direction_id, departure_timestamp)
```

The daily export does not distinguish Vehicle Position stop times from Trip
Update stop predictions after coalescing them. Thirty-nine trips contain equal
exported ordering timestamps, so their 384 rows cannot recover a unique public
order and remain explicitly ineligible for event lowering. For the remaining
rows, exact agreement with the source movement-headway label is the target
eligibility witness; mismatch and boundary classes remain measured masks.

Simultaneous departures retain their separate physical identities but cannot
form one strict `headway` order. Likewise, duplicate aliases collapse only when
trunk and source label agree. `run` relations retain only physical endpoints
with one predecessor and one successor; ambiguous continuations remain counted.

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
| represented source rows / physical departures | 947,714 / 947,489 |
| duplicate / conflicting physical aliases | 225 / 0 |
| ambiguous-order trips / rows | 39 / 384 |
| exact / mismatched / boundary-only source headways | 940,776 / 201 / 1,007 |
| strict headway / unambiguous run relations | 940,752 / 877,168 |
| simultaneous departure groups / events | 206 / 413 |
| ambiguous run sources / targets | 84 / 56 |
| station-trunk-directions | 259 |
| median / p95 / maximum derived gap | 399 / 960 / 57,860 seconds |

The missing identity is not diffuse harmless noise. Examples include every one
of the 5,112 Green-D rows on 2026-08-12 and all 4,276 Green-D rows on
2026-08-14. Removing unresolved rows would remove whole route-days and
preferentially discard atypical operations. Schedule coverage must therefore
be a reported mask, never a population filter.

The daily source exposes event time but no per-event generation, ingestion, or
as-of clock. `index.csv.last_modified` is publication metadata for a completed
file, not fact availability. The current population therefore supports only
retrospective event-time forecasting, regardless of Schedule identity.

## Reproduction

The acquisition commands require explicit temporary or external directories:

```console
uv run --locked -m experiments.tools.mbta_population plan --observed-at 2026-08-22T20:24:04+00:00 --output /tmp/mbta-population-plan.json
uv run --locked -m experiments.tools.mbta_population acquire --plan /tmp/mbta-population-plan.json --source-dir /tmp/mbta-population-source
uv run --locked --with duckdb==1.4.1 -m experiments.tools.mbta_population record --source-dir /tmp/mbta-population-source --output-dir experiments/fixtures/mbta_population
uv run --locked python -m experiments.run mbta_population
```

The corrected clean decision run is bound to revision
`17df7e054df1ddd2ef78e34ea091695428016210`. It reads no study gitlink and
records an empty reference set.

## Stage 3 outcome

Stage 3 froze a [retrospective next-headway task](mbta-headway-task.md) under
five constraints:

- derive inputs and targets only from observed movement events;
- retain added, disrupted, and `NONREV-*`-aliased rows when they satisfy the
  task's explicit movement and lane eligibility rules;
- mask ambiguous order, simultaneous strict-order, mismatched-label, boundary,
  and ambiguous-run cases by their named reason;
- expose exact Schedule identity as a mask and report baseline coverage;
- treat source-provided scheduled values as plan-derived annotations with LAMP
  provenance, never as independently reversible Schedule calls.

A public-plan baseline is reported across every lane where an active Schedule
interval exists. Exact Schedule identity remains a provenance mask and does not
define either that baseline or the target population. Persistence and the
station-local temporal baseline remain independent controls. A future
identity-bearing source can strengthen the Schedule comparison without changing
the event carrier.

## Limits

The audit proves population size, source lineage, Schedule-identity coverage,
and missing availability clocks for one explicit four-week 2026 population.
It does not itself prove a forecast split, baseline score, topology value, or
online-causal claim; Stage 3 owns the first two. Direct GTFS-Realtime archives
or a future MBTA export with feed timestamps could support a stronger
availability contract.
