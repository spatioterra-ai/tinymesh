# Experiments

Experiments decide whether a primitive, component, dataset boundary, or model
claim has earned its place. They do not define the public API.

`experiments.CATALOG` is the source of truth:

```console
uv run --locked python -m experiments.run --list
```

Its groups have one meaning:

- `kernel` measures sparse execution and scaling;
- `primitive` tests a graph operation or an intentionally unpromoted
  composition;
- `layer` proves a reusable `tinymesh.nn` component;
- `data` validates a public loader and its alignment;
- `forecast` compares research policy and predictive evidence.

An owner beginning with `tinymesh.` names the promoted component. An owner of
`research-only` means the result informs a decision without creating an API.
Negative model quality can reject a claim without invalidating a layer whose
equation, gradients, shapes, and sparse work remain correct.

Every stage follows the repository
[closure gate](../docs/experiments.md#close-a-stage) before another dataset or
architecture begins.

## Record a run

Pass experiment settings explicitly:

```console
uv run --locked python -m experiments.run mean_sage DEV=CPU
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU EPOCHS=10 SEED=0
uv run --locked python -m experiments.run transport_forecast DEV=CPU MODEL=diffusion_gru TOPOLOGY=true SEED=0
```

The runner refuses dirty tracked work, executes only cataloged modules, clears
inherited experiment settings, and accepts only the settings declared by that
entry. A successful run writes an ignored JSON envelope under
`experiments/runs/` with:

- the tinymesh revision and every pinned reference gitlink;
- the experiment group and API owner;
- explicit settings, Python version, UTC start, timeout, and elapsed time;
- the experiment's JSON observation.

Local envelopes are working evidence, not release artifacts. A result that
changes a decision belongs in `docs/research/` with its command, revisions,
scope, controls, and limits.

## Graduation gate

A component moves into `src/tinymesh/` only when it has:

1. a stable mathematical contract and a live caller;
2. host or dense parity where applicable;
3. first-order gradient and shape evidence;
4. sparse-work evidence without network-scale dense carriers;
5. a smaller public owner than leaving duplicate implementations in research.

Dataset accuracy governs model claims. It does not, by itself, govern whether a
general component is correctly implemented.
