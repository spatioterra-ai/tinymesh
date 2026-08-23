# MBTA event memory

Status: Stage 2 closed at validation; test was not recomputed.

## Decision

Stop event-memory work on this task. True-topology memory is the best new arm,
but its 147.726-second mean validation MAE loses to the frozen topology MLP's
145.793 seconds in every seed. No runtime API, dependency, or second-caller
study is justified.

```text
completed lane event
        |
        v
  read state at frozen source cutoff
        |
        +---- predict next headway from frozen anchor + memory
        |
        v
  update four-value lane state at departure time
        |
        +---- latest upstream state through 831 sparse edges
                         |
            +------------+------------+------------+
            |            |            |            |
          self          true        reverse      permuted
        151.150       147.726       151.414       150.871 s
                         |
             incumbent topology MLP: 145.793 s
                         |
                  stop at validation
```

The narrower finding is real: directional topology remains identifiable after
adding memory. It does not follow that learned event state improves the
incumbent.

## Causal fold

The experiment reproduces the frozen 940,551-target task, 259 lanes, 831 true
edges, splits, masks, baselines, and topology artifacts before model
construction. Events remain the source of truth; no regular clock or
event-pair graph is built.

For a target whose known source event occurred at `t`, the model reads memory
at the frozen cutoff `t + 1 second`. Completed departures update memory only
when their departure timestamp is strictly earlier than that cutoff. Each
service day starts from empty state, and every train, validation, and
independent replay rebuild does the same. Equal-time events read one old state
and then update as one deterministic group.

This distinction matters because the target departure timestamp is future
information at prediction time. During development, an exploratory run
incorrectly measured message age at that future timestamp; self age then
encoded the target headway and produced an implausible 21.39-second MAE. That
run was rejected before freezing evidence. A pure cutoff regression and an
online-versus-independent-fold invariant now guard the corrected boundary.

In functional terms, service-day replay is a left fold over ordered timestamp
groups:

```text
(memory, group) -> (reads_from_old_memory, advanced_memory)
```

The correspondence stops at training: parameter updates and tinygrad tensor
realization remain an imperative shell.

## Matched model

The experiment-owned encoder is intentionally small:

- a four-value GRU state per lane and two learned cosine elapsed-time features;
- predict-before-update over a fixed 1,024 train lane-day subset;
- one bounded pass, eight-event truncation, and validation selection between
  the unchanged anchor and the trained encoder;
- a matched zero-initialized `27 -> tanh(16) -> 1` residual head for every arm,
  trained for 250 steps over 1,024,000 sampled examples per seed.

The head receives the frozen Stage 4 anchor inputs plus local state and, where
applicable, an affinity-weighted latest upstream state read strictly before the
cutoff. `node_local` removes messages; `self`, `true`, `reverse`, and
degree-preserving `permuted` change only the sparse relation.

This borrows explicit last-update time and event memory from
[TGN](https://arxiv.org/abs/2006.10637), the irregular-observation framing from
[TGNN4I](https://proceedings.mlr.press/v206/oskarsson23a.html), and direct
relation separation from [LRGCN](https://arxiv.org/abs/1905.03994). It does not
port TGN stores, attention, sampling, or link prediction, nor LRGCN's recurrent
relational stack. The exact executable references are pinned PyG
[`5c6461b2`](https://github.com/pyg-team/pytorch_geometric/tree/5c6461b2305ad068a6d61165b3c55852a11aaa41),
PyG Temporal
[`fe555bc3`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415),
and tinygrad
[`33755a34`](https://github.com/tinygrad/tinygrad/tree/33755a34657d25920914badbe32a9d70489669c7).

## Validation evidence

| Arm | Mean MAE | Mean route-macro MAE | Mean p90 AE | Mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| node-local | 151.408 s | 172.091 s | 323.422 s | 364.891 s |
| self | 151.150 s | 171.630 s | 323.880 s | 364.664 s |
| **true** | **147.726 s** | **168.156 s** | **314.649 s** | **363.821 s** |
| reverse | 151.414 s | 171.947 s | 324.586 s | 365.905 s |
| permuted | 150.871 s | 171.824 s | 322.196 s | 364.973 s |

True-topology seed MAEs are 147.712, 148.208, and 147.257 seconds. It beats
every memory control in every seed and on route-macro MAE and p90 error. The
frozen true-topology incumbent reaches 145.742, 146.026, and 145.610 seconds in
the paired seeds, so every promotion clause fails.

Encoder training itself helps only seed 2: seeds 0 and 1 select the unchanged
152.176-second anchor over trained values of 153.507 and 155.693 seconds; seed
2 selects the trained 151.528-second checkpoint. This supports the stop: the
tiny recurrent update is not a stable local improvement.

Every learned arm covers all 138,910 validation targets and reports route,
Schedule-provenance, tail, positivity, parameter, and replay-work evidence.
Persistent state is exactly `259 * 4 = 1,036` values. The true arm processes
313,439 validation edge incidences and 130,351 available messages; storage
remains `O(NH + E)`.

## Evidence lifecycle

Two clean source rebuilds produced byte-identical protocols and identical
encoders, results, baselines, and decisions. Only validation evidence is
retained:

| Artifact | SHA-256 |
| --- | --- |
| protocol | `5db57ecb2f93b1db2ed90afc2d37b41d5c38557f29c21aa6ed901d2df71ebfa9` |
| validation | `9e66ab9d6b7b4808aec222e9280bed7a868cb65c5fcea23abf576de25e2c81ec` |

The validation gate is false, so the already-public confirmatory test split was
not recomputed and no test artifact exists.

```console
uv run --locked --with duckdb==1.4.1 --with numpy==2.3.2 -m experiments.tools.mbta_event_memory --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --task-protocol experiments/fixtures/mbta_headway_task/protocol.json --topology-protocol experiments/fixtures/mbta_topology/protocol.json --topology-validation experiments/fixtures/mbta_topology/validation.json --topology-test experiments/fixtures/mbta_topology/test.json --clock-audit experiments/fixtures/mbta_clock/audit.json --output-dir /tmp/mbta-event-memory
uv run --locked python -m experiments.run mbta_event_memory
```

DuckDB and NumPy remain ephemeral evidence-builder tools. Installed TinyMesh
gains no dependency or public memory abstraction.

## Limits

This is one bounded GRU-style state, one training budget, one retrospective
MBTA interval, and one task. It does not reject event memory generally. It does
show that architecture familiarity is insufficient reason to add it here: the
frozen sparse as-of topology MLP is simpler and better. The event-memory branch
therefore closes until a new caller supplies a distinct residual and its own
causal evidence.
