# Experiments

Experiments decide what tinymesh may claim. They do not own reusable graph
math.

```text
src/tinymesh/      equation + invariants + sparse execution
experiments/       data, training, controls, measurements
docs/research/     revision-bound decisions
```

Paper-derived experiments declare both their pinned paper keys and one fidelity
level in `experiments.CATALOG`:

```text
mechanism     tiny fixture; equation or training path only
ablation      controlled question; departures are explicit
reproduction  same data + splits + preprocessing + model + training + metric
```

Use the smallest level that answers the question. A reproduction must match the
paper protocol end to end; sharing only the dataset is an ablation. The runner
stores this claim beside the revision and settings, so a later result cannot be
silently described at a stronger level.

`experiments.CATALOG` classifies every runnable experiment as a kernel,
primitive, layer, data boundary, or forecast. Its owner names the promoted
`tinymesh` surface or says `research-only`.

List the catalog:

```console
uv run --locked python -m experiments.run --list
```

Record a run by passing every experiment setting explicitly:

```console
uv run --locked --with torch==2.8.0 --with torch-geometric==2.8.0 python -m experiments.run framework_benchmark DEV=METAL DEGREE=8 HIDDEN=32 NODES=4096 SAMPLES=50 WARMUPS=10 WIDTH=32
uv run --locked python -m experiments.run mean_sage DEV=CPU
uv run --locked python -m experiments.run gine DEV=CPU
uv run --locked python -m experiments.run jepa_mechanics DEV=CPU EMA=0.99 HIDDEN=8 LR=0.01 SAMPLES=16 SEED=0 STEPS=80
uv run --locked python -m experiments.run mutag_data DEV=CPU
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=0 STEPS=80 TARGETS=3
uv run --locked --with scikit-learn==1.7.2 python -m experiments.run mutag_graph_jepa_reproduction DEV=METAL
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run metr_la_jepa DEV=METAL BS=64 EMA=0.998 EVAL_SAMPLES=512 HIDDEN=8 HISTORY=12 HORIZON=12 LR=0.001 PROBE_LR=0.01 PROBE_STEPS=100 SAMPLES=512 SEED=0 STEPS=100 TEST=0
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU EPOCHS=10 SEED=0
uv run --locked python -m experiments.run metr_la_forecast DEV=CPU
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL STEPS=3 SEED=0
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=0 BS=512
uv run --locked python -m experiments.run metr_la_diffusion DEV=CPU
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
```

`framework_benchmark` compares steady-state forward paths against PyG 2.8.0
and the exact pinned PyG Temporal `TGCN` source. TGCN includes eager and
full-graph compiled reference paths plus a one-propagation PyTorch control that
matches Tinymesh's factorization. Compilation occurs before timing. The
external packages are resolved only for that command; they are not tinymesh
dependencies. The PyG Temporal 0.56.2 package is not used because it resolves
`torch-sparse` 0.6.18 from source on Python 3.12/macOS; this benchmark needs
only the pinned `TGCN` file.

The runner refuses a dirty tracked worktree, clears inherited experiment
settings, accepts only settings declared by the catalog, and applies the
cataloged positive timeout. The default is 600 seconds; `metr_la_diffusion`
uses a measured 900-second bound. A successful run writes an ignored JSON
envelope under `experiments/runs/` containing:

- the tinymesh revision and all five reference gitlinks;
- the experiment group, API owner, pinned papers, and fidelity level;
- Python version, UTC start, elapsed time, timeout, and explicit settings;
- the experiment's JSON observation.

Local envelopes support iteration. A result that changes a project decision
belongs in `docs/research/` with its exact command, revisions, controls, scope,
and limits.

## Benchmark retention

Benchmarks are retained, not run as routine verification:

```text
tracked runner + protocol    executable question
ignored run envelope         complete local observation
tracked research record      revision-bound result + decision + limits
unit tests                    deterministic benchmark setup, never timing
```

Rerun a benchmark when its implementation, reference pin, hardware, or protocol
changes, or immediately before making a performance claim. Do not add framework
installs or noisy wall-clock thresholds to CI. The current cross-framework
record is the [Mac framework benchmark](research/framework-benchmark.md).

`metr_la_forecast` keeps three modes explicit: no `EPOCHS` or `STEPS` records
the protocol and baselines; `STEPS=N` records bounded optimizer execution
without a quality claim; `EPOCHS=N` runs checkpoint-selected model evidence.
`EPOCHS` and `STEPS` cannot be combined.

`metr_la_diffusion` uses the same bounded execution and evaluation policy with
sequential `DiffusionGRU`, real affinity, and cyclical calendar inputs. Keeping
it as a separate catalog entry preserves the A3T-GCN evidence contract.

`metr_la_local_diffusion` keeps node-local recurrence outside graph propagation
and applies directed transport only through a zero-gated residual. Matched
controls identify a topology signal, but the composition remains research-only
because it does not beat the incumbent.

`metr_la_jepa` masks one causal future block and evaluates frozen factorized
representations against their identical random encoders. The target encoder sees
normalized speed and explicit missingness, never calendar features or test rows;
every arm uses the same fixed 512-window validation sample. `TEST=1` opens the
matching fixed test sample only after validation decides.

Training emits validation evidence only. `TEST=1` explicitly opens final test
evaluation after the head, loss, topology set, seeds, and budget are frozen.
Repeated test queries are development evidence, not an untouched benchmark.

## Promotion

A component enters `src/tinymesh/` when one stable contract has a live caller,
host or dense parity where applicable, first-order gradient and shape evidence,
sparse-work evidence, and a smaller public owner than duplicate research
implementations.

Predictive quality answers a different question. A model may lose to a
baseline while its general graph convolution or recurrent cell remains
correct, sparse, and reusable. The failed result limits the model claim; it
does not move correct math back into an experiment.

No result creates an API automatically. Dataset policy, task readouts,
train/validation/test rules, checkpoint selection, and claims remain
experiment-owned.

Repeated syntax alone is not a promotion gate. Root subtraction and direction
concatenation are ordinary tensor composition with no state or new invariant,
so each model owns that choice directly. A public owner must remove a concept,
not merely a line.

## Close a stage

An experiment stage is complete only when all five boundaries are explicit:

1. **Evidence:** freeze the revision, settings, splits, controls, seeds,
   metrics, and result.
2. **Review:** state exactly what the result proves, does not prove, and what
   evidence would change the decision.
3. **Promotion:** move only repeated, stable mathematical contracts with live
   callers into `src/tinymesh/`; record why everything else remains
   research-only or is deleted.
4. **Distillation:** synchronize the research record, experiment catalog,
   README capability claims, documentation index, and navigation without
   copying local run envelopes into Git.
5. **Delivery:** run the complete relevant verification, merge the coherent
   change, confirm main-branch CI and documentation deployment, then remove the
   branch and worktree.

Do not begin a new dataset or architecture stage while one of these is
unfinished. A negative result closes a claim as completely as a positive one
when its controls and limits are preserved.
