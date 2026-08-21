# MBTA Blue Line replay data

Status: retained Stage 2 boundary evidence; unsuitable for an observed-arrival
forecast target.

## Purpose and status

This replay tests whether real operational transit records preserve enough
identity, time, schedule provenance, and event lineage for a causal TinyMesh
forecast. It proves the replay boundary and rejects the proposed target. It is
not a public dataset API, benchmark, or claim about MBTA service quality.

```text
checksum-pinned LAMP parquet       active Schedule archive rows
              |                               |
              +---------- exact join --------+
                              |
                              v
                  2-hour Blue Line replay
                              |
                   +----------+----------+
                   |                     |
             boundary audit        target lineage
                   |                     |
                   v                     v
             replay is usable      forecast is blocked
```

## Source, license, and citation

The Massachusetts Bay Transportation Authority provides the source data
through its [Lightweight Application for Measuring Performance](https://performancedata.mbta.com/)
(LAMP). The retained source contract uses
[`mbta/lamp@e266440d`](https://github.com/mbta/lamp/tree/e266440db994ed33eede5e44a137b205e4a1e8dd)
and [MBTA GTFS documentation at `02da961b`](https://github.com/mbta/gtfs-documentation/tree/02da961b963ba3d3a66042ca4d5bd19e21ce5c0a).
The exact source URLs, byte counts, and SHA-256 digests live in the fixture
manifest.

Use is subject to the
[MassDOT Developers License Agreement](https://cdn.mbta.com/sites/default/files/developers/2018-10-30-massdot-developers-license-agreement.pdf)
recorded in that manifest. Derived use must acknowledge MassDOT as provider,
comply with law, avoid affiliation or agency claims, avoid MassDOT and MBTA
marks, not misrepresent the data, and not claim ownership. The source is
provided as-is and may change or disappear. TinyMesh is not affiliated with or
endorsed by MassDOT or MBTA.

## Acquisition manifest

The retained slice derives from five exact parquet objects:

- the 2026-08-18 subway performance export;
- the 2026 LAMP `feed_info`, `trips`, `stop_times`, and `stops` archives.

The active schedule is `Summer 2026, 2026-08-17T19:35:03+00:00, version D`,
whose archive applicability window covers 2026-08-18 and 2026-08-19. Raw
parquet remains outside Git. The checked-in manifest pins source and derived
checksums; four CSV tables preserve the transparent projection. Reproduction
uses ephemeral `duckdb==1.4.1`; installed TinyMesh still depends only on
tinygrad.

## Population, axes, and identity

The population is Blue Line revenue rail service on 2026-08-18 from 07:00
through 09:00 America/New_York, equivalent to the half-open UTC interval
`[1787050800, 1787058000)`.

One trip instance is `(service_date, start_time, trip_id)`. One replay row adds
`stop_sequence`; one vehicle is `vehicle_id` within the retained source. Stop
and trip identities resolve exactly to the applicable Schedule rows. Five-minute
intervals are derived from `coalesce(move_timestamp, stop_timestamp)` as
declared by the manifest; they are audit bins, not simultaneous snapshots.

## Missingness and provenance

| Field | Present | Provenance |
| --- | ---: | --- |
| `move_timestamp` | 663 / 663 | observed Vehicle Position progression event |
| `stop_timestamp` | 663 / 663 | mixed: Vehicle Position `STOPPED_AT` or TripUpdate arrival prediction |
| scheduled arrival | 663 / 663 | applicable GTFS Schedule |
| scheduled departure | 663 / 663 | applicable GTFS Schedule |

LAMP documents the stop timestamp fallback but this export does not retain
which source won for a row. Completeness therefore does not establish
observation. The export also omits feed generation and ingestion/as-of clocks,
so observation age is unavailable. Neither fact is imputed.

## Transformations and splits

The source projection selects one route, service date, and two-hour interval,
then joins each trip-stop row to the applicable Schedule archive. It performs
no deduplication, aggregation, normalization, tensor lowering, train/validation
split, or target construction. CSV row order is canonical trip-instance and
stop-sequence order.

## Validation and statistics

| Measure | Result |
| --- | ---: |
| replay rows / resolved schedule rows | 663 / 663 |
| trip instances / vehicles / stops | 66 / 12 / 24 |
| duplicate trip-stop identities in slice | 0 |
| five-minute intervals | 24, all populated |
| rows per interval | 25–30 |
| directed Schedule-union edges | 22 |
| directed edges active in replay | 22 |
| active edges per interval | 21–22 |
| `(interval, stop)` groups with multiple vehicles | 106 |
| maximum vehicles in one such group | 2 |

The full-day source contains 41,446 rows, 3,722 trip instances, 241 vehicles,
2,242 missing movement timestamps, 583 missing stop timestamps, and 59 duplicate
trip-stop groups. Those source-level facts explain why collision and missingness
policies cannot be inferred from this unusually complete slice.

## Intended use

Use this fixture to test acquisition drift, schedule alignment, identity,
missingness, provenance, interval coverage, and sparse topology activity. It is
also a concrete input for designing a future replay contract with source-tagged
events.

## Prohibited claims

Do not call `stop_timestamp` an observed arrival, derive observed delay from it,
interpret audit bins as simultaneous vehicle state, report observation
freshness, train or evaluate a forecast target from these mixed timestamps, or
generalize this two-hour slice to MBTA performance.

## Limits

This is one line, one weekday, and two hours. The retained schedule is only the
66 selected trips, not the full MBTA network. Platform-level direction produces
22 directed edges. The interval collision count shows that node aggregation is
material, but without a defensible target there is no reason to freeze an
aggregation or active-versus-union topology policy.

## Evidence

```console
uv run --locked python -m experiments.run gtfs_replay
```

The executable audit validates all retained artifact checksums, typed fields,
canonical identities, interval bounds, exact Schedule joins, event lineage,
coverage, topology activity, and vehicle collisions using the Python standard
library. Stage 3 is blocked: this source has zero source-tagged observed-arrival
targets. Per the epic stop rule, that negative result adds no model or public
API.
