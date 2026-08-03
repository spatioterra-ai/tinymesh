# Experiments

Experiments decide what tinymesh may claim. They do not own reusable graph
math.

```text
src/tinymesh/      equation + invariants + sparse execution
experiments/       data, training, controls, measurements
docs/research/     revision-bound decisions
```

`experiments.CATALOG` classifies every runnable experiment as a kernel,
primitive, layer, data boundary, or forecast. Its owner names the promoted
`tinymesh` surface or says `research-only`.

List the catalog:

```console
uv run --locked python -m experiments.run --list
```

Record a run by passing every experiment setting explicitly:

```console
uv run --locked python -m experiments.run mean_sage DEV=CPU
uv run --locked python -m experiments.run gine DEV=CPU
uv run --locked python -m experiments.run jepa_mechanics DEV=CPU EMA=0.99 HIDDEN=8 LR=0.01 SAMPLES=16 SEED=0 STEPS=80
uv run --locked python -m experiments.run mutag_data DEV=CPU
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=0 STEPS=80 TARGETS=3
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU EPOCHS=10 SEED=0
uv run --locked python -m experiments.run metr_la_forecast DEV=CPU
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL STEPS=3 SEED=0
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=0 BS=512
uv run --locked python -m experiments.run metr_la_diffusion DEV=CPU
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
```

The runner refuses a dirty tracked worktree, clears inherited experiment
settings, accepts only settings declared by the catalog, and applies the
cataloged positive timeout. The default is 600 seconds; `metr_la_diffusion`
uses a measured 900-second bound. A successful run writes an ignored JSON
envelope under `experiments/runs/` containing:

- the tinymesh revision and all five reference gitlinks;
- the experiment group and API owner;
- Python version, UTC start, elapsed time, timeout, and explicit settings;
- the experiment's JSON observation.

Local envelopes support iteration. A result that changes a project decision
belongs in `docs/research/` with its exact command, revisions, controls, scope,
and limits.

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
