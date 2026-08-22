# MBTA next-headway task

Status: Stage 3 frozen; test opened once after validation freeze.

## Decision

Freeze a retrospective next-headway task over the full exact movement-derived
population. The task is large, deterministic, and usable. Its limit is not data
quality but availability: LAMP has event time and no generation or ingestion
clock, so this evidence cannot support an online claim.

```text
sealed 28-day LAMP + active Schedule
                 |
                 v
       947,489 physical departures
                 |
          exact-label mask
                 v
       940,551 physical targets
          /          |          \
  train 625,073  val 138,910  test 176,568
          |          |          |
       fit only   select once   open once
          \          |          /
           +---- frozen task ---+
                        |
                        v
               Stage 4 model study
```

The 940,551 targets reconcile with Stage 2's 940,776 exact source labels: 225
rows are duplicate aliases of already represented physical departures. No
conflicting alias is present.

## Prediction contract

For consecutive strict departure times `a` and `b` in one
`(service_date, parent_station, trunk_route_id, direction_id)` lane:

```text
cutoff = time(a) + 1 second
target = time(b) - time(a)
inputs = events whose timestamp < cutoff
```

The one-second cutoff includes every event simultaneous with `a` and excludes
every event at `b`. A target is retained only when the movement-derived interval
equals the source headway. Mismatched and boundary-only source labels therefore
remain named population masks, not training targets. Ambiguous public order,
simultaneous strict order, ambiguous run endpoints, duplicate aliases, and
unresolved Schedule identity also remain separately measured; none silently
changes physical identity.

The split is by **target service date**, never by row:

| Split | Inclusive dates | Days | Targets |
| --- | --- | ---: | ---: |
| train | 2026-07-24–2026-08-10 | 18 | 625,073 |
| validation | 2026-08-11–2026-08-15 | 5 | 138,910 |
| test | 2026-08-16–2026-08-20 | 5 | 176,568 |

Green-C has no validation target in this observed interval. Route macro metrics
therefore average the routes present in that split and never impute an absent
route-day.

## Baselines and leakage boundary

Three controls define the Stage 4 floor:

- **persistence:** the preceding exact headway ending at `a`;
- **temporal:** train-fitted median with fallback
  `station-lane × weekday × hour -> lane -> route -> global`;
- **plan:** the next active public Schedule interval at or after the cutoff in
  the same service-day lane.

The temporal search is limited to hour bins `{1, 2, 4}` and minimum supports
`{4, 16, 64}`. Only validation MAE selects the candidate. Schedule intervals
come from active GTFS calendar, exception, trip, stop, and call records; the
target row's scheduled fields are never read. Exact Schedule identity is
reported only as a coverage mask. Because GTFS version activity and LAMP event
availability are date-level and retrospective, the plan result is a public-plan
control, not proof of information available to a live predictor.

This mirrors the useful part of TSL's temporal splitter and train-slice fitting:
forward partitions share one mask while learned statistics see train only.
LibCity motivates the familiar MAE/RMSE report, but TinyMesh aggregates raw
errors globally rather than averaging batch summaries. PyG Temporal's static
snapshot container is retained only as a future clock control; it cannot own
the event-time task.

## Frozen evidence

Primary selection metric is micro MAE over every covered target. RMSE,
median/p90 absolute error, per-route MAE and count, macro-route MAE, and coverage
are retained diagnostics.

| Baseline | Coverage | MAE | RMSE | Median AE | p90 AE | Macro-route MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 99.3233% | 217.969 s | 400.224 s | 145 s | 481 s | 239.623 s |
| temporal median | 100% | 157.152 s | 365.044 s | 103 s | 336 s | 175.667 s |
| public plan | 99.9028% | **152.648 s** | 382.367 s | 99 s | 323 s | **174.804 s** |

Validation selects the one-hour, minimum-support-four temporal control. It uses
the full cell estimate for 137,670 targets and lane fallback for 1,240; route
and global fallback are unused. The public-plan control wins MAE but has worse
RMSE than the temporal median, identifying a tail worth preserving rather than
collapsing the decision to one score.

Plan coverage is 89,847/89,847 on Schedule-resolved targets and 48,928/49,063
on unresolved target aliases. This is expected: the baseline is a lane-level
public plan and does not require the observed vehicle to resolve to a scheduled
trip. The mask measures provenance, while the baseline measures whether a lane
plan exists at cutoff.

The test split was opened once after commit `c44f33a39d`. The selected temporal
configuration was not revised:

| Baseline | Coverage | MAE | RMSE | Median AE | p90 AE | Macro-route MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 99.3028% | 229.802 s | 437.388 s | 151 s | 512 s | 229.647 s |
| temporal median | 100% | **159.788 s** | **346.368 s** | **102 s** | 353 s | **158.356 s** |
| public plan | 99.9037% | 166.062 s | 351.393 s | 112 s | **352 s** | 165.392 s |

The validation winner does not remain the test winner. The public plan regresses
by 13.414 s MAE while the temporal control changes by 2.636 s. Stage 4 must beat
both controls rather than treating either split-specific winner as sufficient.

## Freeze and reproduction

The retained `protocol.json` binds the sealed source manifest, Stage 2 audit,
split, masks, counts, metrics, seeds, and reference gitlinks. Validation binds
the protocol checksum. The test command first rebuilds and byte-compares both
frozen artifacts, refuses an existing `test.json`, and only then evaluates test.

```console
uv run --locked --with duckdb==1.4.1 -m experiments.tools.mbta_headway_task --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --output-dir /tmp/mbta-headway-task
uv run --locked python -m experiments.run mbta_headway_task
```

To reproduce the already recorded single test opening in the fresh output is:

```console
uv run --locked --with duckdb==1.4.1 -m experiments.tools.mbta_headway_task --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --output-dir /tmp/mbta-headway-task --test
uv run --locked python -m experiments.run mbta_headway_task TEST=1
```

The executable record pins TSL `aa5f313e000d192bdec270748b8d01df5912e58e`,
LibCity `5a6391d41944e937f2c15e9be85ab7f40ac8b23e`, and PyG Temporal
`fe555bc30ee197755c4b58a89407033a5f383415`. They are study references, not
runtime dependencies.

## Stage 4 consequence

Stage 4 receives this carrier, cutoff, split, masks, baseline implementations,
and metrics unchanged. It should begin with a causal event-prefix model and a
matched clock control, require improvement over the stronger of public plan and
temporal median, report mask slices and per-route counts, and keep all reusable
API decisions closed until the experiment demonstrates an identifiable gain.

## Limits

This task covers one 28-day summer interval and one transit system. It has no
weather, incidents, crowding, or true as-of ingestion clock. Validation chooses
one baseline configuration; test will estimate generalization once, not rescue
or revise it. Strong performance here will remain retrospective MBTA evidence,
not a universal transit or production-readiness claim.
