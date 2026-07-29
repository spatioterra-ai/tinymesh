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

## Protocol

Revision `aa8e30030055b9996217f103d1a5a775f3a4a655` generated 128 training,
32 validation, and 32 test trajectories from host-random seeds `20260729`,
`20260730`, and `20260731`. Every trajectory contains eight steps on 24 nodes
and 36 directed edges. Splits therefore differ by complete initial fields, not
overlapping windows.

The matched recurrent comparison used hidden width 8, batch size 64, Adam at
`0.01`, 30 epochs, validation-RMSE checkpoint selection, and model seeds 0, 1,
and 2. Each model saw four steps. Evaluation reports all teacher-forced
one-step windows and one four-step recursive rollout per trajectory.

```console
uv run --locked python -m experiments.run transport_forecast DEV=CPU MODEL=diffusion_gru TOPOLOGY=true SEED=0 EPOCHS=30 HISTORY=4 HORIZON=4 BS=64 HIDDEN=8 LR=0.01
```

Repeat with `SEED=1` and `SEED=2`; replace `TOPOLOGY=true` with
`TOPOLOGY=permuted` and `TOPOLOGY=self` for the structural controls.
`MODEL=gconv_gru` accepts true or permuted symmetric support, while
`MODEL=lstm TOPOLOGY=none` is node-local. The three-parameter linear diagnostic
used the same protocol with `MODEL=diffusion_linear` and `EPOCHS=100`.

## Results

Mean RMSE plus sample standard deviation across three seeds is:

| Model | Topology | Parameters | Validation one-step | Validation rollout | Test one-step | Test rollout |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | none | 0 | 0.037898 | 0.097644 | 0.038644 | 0.098729 |
| LSTM | none | 361 | 0.024328 ± 0.002547 | 0.058249 ± 0.002221 | 0.025785 ± 0.002708 | 0.061232 ± 0.002349 |
| GConvGRU | true symmetric | 465 | 0.015028 ± 0.001703 | 0.034017 ± 0.001147 | 0.015371 ± 0.001584 | 0.033789 ± 0.000993 |
| GConvGRU | permuted symmetric | 465 | 0.015518 ± 0.001026 | 0.035142 ± 0.001339 | 0.016913 ± 0.001334 | 0.037026 ± 0.001887 |
| DiffusionGRU | true | 681 | **0.003030 ± 0.000802** | **0.004232 ± 0.001222** | **0.003143 ± 0.000905** | **0.004442 ± 0.001385** |
| DiffusionGRU | permuted | 681 | 0.015317 ± 0.002695 | 0.020448 ± 0.001736 | 0.015697 ± 0.002533 | 0.020691 ± 0.001704 |
| DiffusionGRU | self | 681 | 0.022408 ± 0.000189 | 0.056434 ± 0.000242 | 0.023684 ± 0.000142 | 0.059339 ± 0.000123 |

On validation, true-topology DiffusionGRU reduces one-step RMSE by 80.2% and
rollout RMSE by 79.3% relative to the degree-matched permutation. It wins both
paired comparisons in all three seeds. Relative to the node-local LSTM, the
reductions are 87.5% and 92.7%, also with three paired wins. Test repeats every
paired win.

The longer linear diagnostic isolates the operator from recurrence:

| Topology | Validation one-step | Validation rollout | Test one-step | Test rollout |
| --- | ---: | ---: | ---: | ---: |
| true | **0.001101** | **0.001885** | **0.001162** | **0.001983** |
| permuted | 0.014877 | 0.023220 | 0.016031 | 0.025086 |

The true graph wins every paired linear comparison. The generic symmetric
GConvGRU has a much smaller separation: true support wins validation one-step
in two seeds and rollout in three. That is consistent with a process whose
direction and scalar affinity matter, but it is not a controlled architecture
ablation.

## Decision

The graph claim passes only if the true-topology model beats persistence and
the node-local model, while also beating its degree-matched false-topology
control on validation one-step and rollout RMSE across at least two of three
seeds. Test data confirms a validation decision; it does not select one.

The gate passes. The existing sparse recurrent path can exploit identifiable
graph signal, and the earlier negative datasets are not evidence that
`DiffusionGRU` ignores topology.

## Limits

This is the easiest honest positive control: the host generator and
`DirectedDiffusion` implement the same stated transport equation independently.
The process is deterministic, fixed-topology, scalar, noiseless, fully
observed, and only four rollout steps long. Its units and topology are
synthetic. Configuration was developed with test output visible, so this is an
engineering witness rather than an untouched benchmark.

The result proves neither a new architecture nor performance on a real
physical system. The next evidence should preserve the same topology controls
on a real sensor network such as METR-LA.
