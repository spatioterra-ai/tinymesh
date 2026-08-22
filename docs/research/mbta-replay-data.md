# MBTA Blue Line replay data

Status: retained boundary evidence; arrival, travel-time, and dwell targets are
rejected; movement-derived trunk headway is validated for replay extension.

## Purpose and status

This replay tests whether real transit operations preserve enough identity,
time, Schedule provenance, and event lineage for a causal TinyMesh task. It is
not a public dataset API, benchmark, or claim about MBTA service quality.

```text
checksum-pinned LAMP performance + active Schedule
                         |
                         v
               2-hour Blue Line replay
                         |
             +-----------+-----------+
             |                       |
       mixed stop time         observed move events
             |                       |
             v                       v
 arrival / travel / dwell      departure headway
          rejected                 viable
```

The first audit correctly rejected observed arrival. A follow-up trace of
LAMP's pinned implementation found that the rejection does not apply to every
calculated field: trunk headway uses Vehicle Position movement events only.

## Source, license, and citation

The Massachusetts Bay Transportation Authority provides the source through its
[Lightweight Application for Measuring Performance](https://performancedata.mbta.com/)
(LAMP). The retained contract uses
[`mbta/lamp@e266440d`](https://github.com/mbta/lamp/tree/e266440db994ed33eede5e44a137b205e4a1e8dd)
and [MBTA GTFS documentation at `02da961b`](https://github.com/mbta/gtfs-documentation/tree/02da961b963ba3d3a66042ca4d5bd19e21ce5c0a).
Exact URLs, byte counts, and SHA-256 digests live in the fixture manifest.

Use is subject to the
[MassDOT Developers License Agreement](https://cdn.mbta.com/sites/default/files/developers/2018-10-30-massdot-developers-license-agreement.pdf)
recorded there. Derived use must acknowledge MassDOT as provider, comply with
law, avoid affiliation or agency claims, avoid MassDOT and MBTA marks, not
misrepresent the data, and not claim ownership. The source is provided as-is
and may change or disappear. TinyMesh is not affiliated with or endorsed by
MassDOT or MBTA.

## Acquisition manifest

The retained slice derives from the 2026-08-18 subway performance export and
the LAMP 2026 `feed_info`, `trips`, `stop_times`, and `stops` archives. Its
active Schedule is `Summer 2026, 2026-08-17T19:35:03+00:00, version D`.

Raw parquet remains outside Git. The checked-in manifest pins source and
derived checksums; four CSV tables preserve the transparent projection.
Reproduction uses ephemeral `duckdb==1.4.1`; installed TinyMesh still depends
only on tinygrad.

## Population, axes, and identity

The population is Blue Line revenue rail service on 2026-08-18 from 07:00
through 09:00 America/New_York, the half-open UTC interval
`[1787050800, 1787058000)`.

One trip instance is `(service_date, start_time, trip_id)`. One replay row adds
`stop_sequence`; one vehicle is `vehicle_id`. Trip-stop identities resolve to
the applicable Schedule. A physical departure is
`(service_date, vehicle_id, parent_station, direction_id, departure_timestamp)`.
This identity collapses duplicate source aliases without collapsing distinct
vehicles or directions.

## Missingness and provenance

| Field | Present | Provenance | Target decision |
| --- | ---: | --- | --- |
| movement timestamp | 663 / 663 | observed Vehicle Position | event input |
| stop timestamp | 663 / 663 | Vehicle Position stop or TripUpdate prediction | reject |
| travel time | 663 / 663 | mixed stop minus observed move | reject |
| dwell time | 553 / 663 | next observed move minus mixed stop | reject |
| trunk headway | 608 / 663 | successive observed departure movements | retain |
| scheduled trunk headway | 663 / 663 | Schedule-derived | baseline candidate |

The export omits feed generation and ingestion/as-of clocks, so it cannot
support a claim about live snapshot freshness. Historical headway has its own
explicit target time: the next stop row's observed `move_timestamp`, which
marks departure from the current stop. A future task must use only events
strictly before that timestamp as input.

## Target derivation

At the pinned LAMP revision:

```text
stop       = coalesce(vp_stop, trip_update_arrival)
travel     = stop - move
dwell      = next(move) - stop
departure  = next(move)
headway    = departure - previous_departure
```

Only the last two expressions avoid the mixed stop value. LAMP partitions
trunk headway by parent station, trunk route, and direction and retains only
positive values. The retained audit independently applies that formula to
adjacent Schedule calls and compares exact integers.

## Transformations and splits

The projection selects one route, service date, and two-hour interval, then
joins each trip-stop row to the applicable Schedule. It performs no source
deduplication, normalization, tensor lowering, train/validation split, or model
aggregation. CSV order is canonical trip-instance and stop-sequence order.

The audit derives departure time from the following adjacent trip-stop movement
event. It groups physical departures by station, trunk route, and direction,
then differences consecutive target times. Values requiring an event outside
the cut window remain boundary-only rather than being imputed.

## Validation and statistics

| Measure | Result |
| --- | ---: |
| replay rows / resolved Schedule rows | 663 / 663 |
| trip instances / vehicles / platform stops | 66 / 12 / 24 |
| parent stations / directed Schedule edges | 12 / 22 |
| source / internally derived headways | 608 / 575 |
| exact headway matches / mismatches | 575 / 0 |
| source labels needing an outside-window event | 33 |
| physical departures / duplicate aliases | 597 / 0 |
| five-minute headway target bins | 485 |
| bins with two targets / maximum targets | 90 / 2 |

The complete 2026-08-18 Blue Line day has 4,706 rows and 4,274 source
headways. All 4,274 reproduce exactly. A bounded seven-day source audit from
2026-08-12 through 2026-08-18 found 26,559 row labels representing 26,558
physical departures. Every physical event reproduced exactly; the one extra
row was the same departure represented under two `ADDED-*` trip IDs with the
same 498-second label. Issue
[#150](https://github.com/spatioterra-ai/tinymesh/issues/150) retains the exact
source checksums and interpretation.

## Intended use

Use this fixture to test acquisition drift, Schedule alignment, identity,
missingness, provenance, topology activity, physical-event deduplication, and
movement-derived headway. It justifies extending the replay before specifying
a headway forecast; it does not itself justify training.

## Prohibited claims

Do not call `stop_timestamp` an observed arrival; treat travel time or dwell as
pure observation; interpret audit bins as simultaneous vehicle state; report
live freshness; generalize this two-hour slice to MBTA performance; or call
LAMP's formula the uniquely correct operational definition of headway.

## Limits

This is one line, one weekday, and two hours. The retained Schedule contains
only the selected trips. Five-minute target bins contain two departures in 90
cases, so a future task must choose event-level prediction, a smaller clock, or
an explicit aggregation. The upstream source calls its headway calculation
incomplete and has no focused test for the metric. TinyMesh therefore claims
only exact reproduction and observational lineage of the pinned formula.

## Evidence

```console
uv run --locked python -m experiments.run gtfs_replay
```

The standard-library audit validates artifact checksums, typed fields,
canonical identities, exact Schedule joins, field-specific lineage, topology,
and headway reconstruction. The next justified step is a longer replay and
task specification for movement-derived trunk headway—not a public adapter or
model added from this two-hour witness.
