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

The retained MBTA replay is reproduced from external checksum-pinned parquet
with an ephemeral extraction tool; DuckDB is not an installed dependency:

```console
uv run --locked --with 'duckdb==1.4.1' --module experiments.tools.mbta_replay_extract \
  --source-dir /path/to/sources \
  --output-dir experiments/fixtures/mbta_replay
```
