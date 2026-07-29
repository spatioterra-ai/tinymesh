# Controlled transport forecast

This experiment asks one narrow question: can the existing recurrent graph
models use the correct topology when the target is known to depend on spatial
transport?

It is a small Tinymesh witness inspired by the
[PDE temporal-graph benchmark](https://openreview.net/forum?id=EguDBMechn) and
its pinned [PyG Temporal
loader](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/dataset/advection_diffusion.py).
It does not reproduce that benchmark's 400-region German graph or source
files.

## Data

An independent host loop evolves zero-mean scalar fields over a fixed directed
24-node graph. Each step is:

```text
x[t+1] = 0.25 x[t] + 0.55 forward(x[t]) + 0.20 reverse(x[t])
```

The coefficients sum to one, so the process conserves total field mass.
Forward and reverse propagation use positive edge affinity normalized at the
source and destination respectively. The generator calls neither `Graph` nor a
tinymesh neural-network component.

Training, validation, and test contain independent initial fields selected by
fixed host-random seeds. A model sees four snapshots and predicts the next one.
Evaluation reports teacher-forced one-step error and a four-step recursive
rollout.

## Controls

The node-local LSTM and persistence floor receive no topology. Graph models
receive either the true graph, a false graph produced by one bijective relabeling
that preserves the complete degree sequence, or self edges only.

Reversing every edge is not a valid control for `DiffusionGRU`: its basis
already contains both forward and reverse propagation, so reversal only swaps
two learnable columns.

The exact run settings and results will be frozen here after the first matched
three-seed comparison.

```console
uv run --locked python -m experiments.run transport_forecast DEV=CPU MODEL=diffusion_gru TOPOLOGY=true SEED=0 EPOCHS=30 HISTORY=4 HORIZON=4 BS=64 HIDDEN=8 LR=0.01
```

## Decision gate

The graph claim passes only if the true-topology model beats persistence and
the node-local model, while also beating its degree-matched false-topology
control on validation one-step and rollout RMSE across at least two of three
seeds. Test data confirms a validation decision; it does not select one.

Passing proves that the current sparse recurrent path can exploit identifiable
graph signal. It does not establish performance on a real physical system.
