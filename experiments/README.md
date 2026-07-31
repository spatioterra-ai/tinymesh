# Experiments

Experiments own revision-bound policy, controls, training, and measurements.
They use the public runtime; they do not define it.

```text
experiments.CATALOG   runnable inventory, settings, timeout, API owner
experiments.run       clean-revision execution and ignored local envelope
docs/research/        durable result and current decision
```

List the exact inventory:

```console
uv run --locked python -m experiments.run --list
```

An owner beginning with `tinymesh.` names an already-promoted contract.
`research-only` means the result creates no API. The canonical
[experiment guide](../docs/experiments.md) owns run syntax, evidence envelopes,
the graduation gate, and stage closure.
