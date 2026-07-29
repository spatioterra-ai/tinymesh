# Controlled transport transfer

This experiment asks whether one model trained on the controlled 24-node
transport graph learns a reusable local operator rather than one fixed network.

The training data, model, optimizer, and validation checkpoint follow the
[controlled transport forecast](transport-forecast.md). After training, the
weights are frozen and evaluated without retraining on:

```text
24, 32, and 48 nodes
        x
dense fields and four localized pulses
        x
16 recursive forecast steps
```

`DiffusionGRU` uses the true, degree-preserving permuted, and self-only
topologies for every scope. The node-local LSTM and persistence floor receive
no graph. One state digest before and after evaluation verifies that transfer
does not mutate trained parameters.

The exact settings, three-seed results, promotion audit, and limits will be
frozen here after the matched study.

```console
uv run --locked python -m experiments.run transport_transfer DEV=CPU MODEL=diffusion_gru NODES=all INITIAL=dense SEED=0 EPOCHS=30 HISTORY=4 HORIZON=16 BS=64 HIDDEN=8 LR=0.01
```

Repeat with `INITIAL=pulse` and model seeds 1 and 2. Each process trains once
and evaluates all three node sizes for one initial-condition family, remaining
below the shared 600-second limit.

The node-local LSTM uses the same settings with `MODEL=lstm` and one explicit
`NODES=32` or `NODES=48` scope per run; its all-size evaluation exceeds that
limit.

The transfer claim passes only if true-topology DiffusionGRU beats its permuted
and self-only controls plus the node-local LSTM on validation-selected,
held-out scopes for both one-step and 16-step rollout RMSE in at least two of
three seeds. The 32- and 48-node scopes must agree; the 24-node scope alone
cannot pass the gate.
