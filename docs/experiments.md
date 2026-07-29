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
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU EPOCHS=10 SEED=0
```

The runner refuses a dirty tracked worktree, clears inherited experiment
settings, accepts only settings declared by the catalog, and bounds execution
at 600 seconds. A successful run writes an ignored JSON envelope under
`experiments/runs/` containing:

- the tinymesh revision and all five reference gitlinks;
- the experiment group and API owner;
- Python version, UTC start, elapsed time, timeout, and explicit settings;
- the experiment's JSON observation.

Local envelopes support iteration. A result that changes a project decision
belongs in `docs/research/` with its exact command, revisions, controls, scope,
and limits.

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

