# METR-LA factorized JEPA

This stage asks whether causal latent prediction improves a reusable traffic
representation—and whether correct spatial transport and learned temporal
mixing account for the improvement. The registered result is negative.

## Causal boundary

The existing METR-LA protocol supplies five-minute traffic speed over 207
sensors and a fixed directed affinity graph. Targets retain the forward
70/10/20 split and train-only standardization.

```text
available history                         hidden future
[12, sensor, speed + observed]      [12, sensor, speed + observed]
              |                                  |
              v                                  v
       online encoder                     EMA target encoder
              |                                  |
              +------ predictor -- L1 -- stop gradient
```

Self-supervision and frozen-probe fitting see training windows only. Calendar
features are excluded from both encoders, so time of day cannot solve latent
prediction without reading the traffic field. Missing speed is zero after
normalization and remains distinguishable through its explicit observed mask.
No test row is read unless `TEST=1`.

## Factorized encoder

```text
[B, 12, N, 2]
      |
      +-- directed diffusion over N, batched across B and time
      |      root + tanh(gate) * projected neighbor residual -> LayerNorm
      |
      +-- one bias-free tinygrad LSTM, shared across nodes
      |      12 causal recurrent steps -> LayerNorm
      |
      v
[B, N, 12 * hidden] -> block predictor
```

Space remains sparse. One encoder call performs one forward and one reverse
CSR reduction over every batch and time row together. Time mixing is bounded
node-local recurrence over 12 periods; it never constructs dense node pairs or
a `node x time` product graph. Bias-free spatial and temporal projections
preserve the zero-field invariant.

The spatial gate starts at zero. `factorized`, `permuted`, and `temporal`
therefore have identical random representations; training alone can introduce
a graph effect. Forward and reverse messages enter as deviations from the root,
so self-loop transport is exactly the node-local control.

The online encoder and two-layer predictor receive gradients. The target
encoder starts as an exact copy, receives no gradient, and follows the online
encoder with EMA `0.998`. Latent L1 compares the predicted 12-step block with
the detached target block.

## Matched arms

| Arm | Spatial operator | Causal time mixer |
| --- | --- | --- |
| `factorized` | true directed affinity | yes |
| `permuted` | degree-matched false graph | yes |
| `temporal` | self loops only | yes |
| `spatial` | true directed affinity | no |

The first three arms have identical parameters and initialization. `factorized`
versus `permuted` tests the particular neighbors, while `factorized` versus
`temporal` tests whether any neighbor messages help. `factorized` versus
`spatial` isolates learned mixing inside the encoder; every probe still sees
all 12 encoded history rows.

## Frozen evaluation

Each arm is probed before and after pretraining with the same initialized
linear head, training windows, optimizer steps, and observed-target mask. A
raw-history linear probe and persistence use the same sampled evaluation rows.
RMSE in miles per hour is primary; MAE and 15-, 30-, and 60-minute slices are
diagnostic.

The fixed budget is:

- 512 seeded training windows for every probe;
- 512 seeded validation windows shared by every arm and reference;
- 512 seeded test windows, opened only by the registered gate;
- batch size 64, hidden width 8, 100 JEPA steps, and 100 probe steps;
- formal seeds 0, 1, and 2.

Each evaluation sample contains up to `512 * 207 * 12 = 1,271,808`
sensor-horizon targets; missing targets are excluded explicitly. Validation and
test samples use fixed data seeds independent of model seed and arm.

## Registered decision

Seed `17` was used before registration to verify execution and reject two
confounded designs: direct neighbor concatenation made random encoders depend
on topology, while a static causal average discarded temporal detail. It is
excluded from evidence, and no formal seed or test row informed the final
design. Formal validation passes only when all mechanics hold and each
architectural direction wins on mean RMSE and in at least two of three paired
seeds:

1. every target gradient is exactly zero, every target moves, and final latent
   variation across training windows remains at least half its initial value;
2. trained `factorized` beats its identical random encoder;
3. trained `factorized` beats trained `permuted` and trained `temporal`;
4. trained `factorized` beats trained `spatial`.

Only a full validation pass opens the fixed test sample. Test must repeat the
same paired directions; it never selects an arm or setting. A passing result
earns one matched supervised-initialization experiment, not API promotion. A
failed gate closes that claim without changing seeds, arms, budgets, features,
or thresholds.

## Paper grounding

[TS-JEPA v1](https://arxiv.org/abs/2509.25449v1) supplies time patches, an
online encoder and predictor, a stop-gradient EMA target, latent L1, frozen
evaluation against the same random encoder, and EMA `0.998`. This experiment
replaces its transformer with sparse directed diffusion plus a bounded causal
time mixer, fixes one future-block mask, and evaluates a multivariate sensor
graph. It is an architectural test, not a TS-JEPA reproduction. Exact paper and
source revisions are pinned by the [paper registry](../papers.md).

## Decision

At revision
[`17441a9`](https://github.com/spatioterra-ai/tinymesh/tree/17441a9ac70b6c19ba7ad3b012b95a0802927f19),
three Metal runs evaluated the same 1,189,650 observed validation targets per
seed. Values are mean and population standard deviation across seeds. Gain is
paired random-encoder RMSE minus trained-encoder RMSE, so positive is better.

| Arm | Random RMSE | Trained RMSE | Gain | Seed wins | EMA target gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `factorized` | 7.523661 ± 0.057822 | 7.616622 ± 0.095102 | **-0.092961 ± 0.047189** | **0/3** | 0.000641 ± 0.000107 |
| `permuted` | 7.523661 ± 0.057822 | 7.593895 ± 0.082578 | -0.070234 ± 0.029370 | 0/3 | 0.000726 ± 0.000076 |
| `temporal` | 7.523661 ± 0.057822 | 7.528295 ± 0.059672 | -0.004633 ± 0.008132 | 1/3 | 0 |
| `spatial` | 7.416151 ± 0.069797 | **7.416944 ± 0.069456** | -0.000792 ± 0.000896 | 1/3 | 0.000080 ± 0.000005 |

The raw-history probe reached `7.281532 ± 0.026356` RMSE and persistence
reached `7.608212`. The trained factorized representation was worse than both.
It also lost to trained `permuted` in two seeds and to `temporal` and `spatial`
in all three; its mean RMSE deficits were `0.022727`, `0.088327`, and
`0.199678`, respectively.

Mechanics are sound but utility fails. Mean factorized latent L1 fell from
`0.790422` to `0.192732`, every target gradient was exactly zero, the target
moved by `1.204118 ± 0.154112`, and representation variation increased from
`0.498568 ± 0.033605` to `0.516190 ± 0.026315`. Low latent loss therefore did
not produce a better frozen traffic representation, and the tiny learned
spatial gates supplied no stable true-topology advantage.

The representation, topology, and temporal-mixer gates fail. Test remains
unopened and the supervised-initialization follow-up does not run. This rejects
this encoder, causal block mask, objective, and budget on METR-LA; it does not
show that traffic lacks graph structure or that every JEPA formulation fails.
Promote no encoder, predictor, objective, probe, or orchestration API. The
root-relative transport repeated from the earlier local-diffusion study is now
`DirectedDiffusion.residual`; that distills shared graph math, not JEPA.

## Reproduce

```console
uv run --locked python -m experiments.run metr_la_jepa DEV=METAL BS=64 EMA=0.998 EVAL_SAMPLES=512 HIDDEN=8 HISTORY=12 HORIZON=12 LR=0.001 PROBE_LR=0.01 PROBE_STEPS=100 SAMPLES=512 SEED=0 STEPS=100 TEST=0
uv run --locked python -m experiments.run metr_la_jepa DEV=METAL BS=64 EMA=0.998 EVAL_SAMPLES=512 HIDDEN=8 HISTORY=12 HORIZON=12 LR=0.001 PROBE_LR=0.01 PROBE_STEPS=100 SAMPLES=512 SEED=1 STEPS=100 TEST=0
uv run --locked python -m experiments.run metr_la_jepa DEV=METAL BS=64 EMA=0.998 EVAL_SAMPLES=512 HIDDEN=8 HISTORY=12 HORIZON=12 LR=0.001 PROBE_LR=0.01 PROBE_STEPS=100 SAMPLES=512 SEED=2 STEPS=100 TEST=0
```
