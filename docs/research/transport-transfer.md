# Controlled transport transfer

This experiment asks whether one model trained on the controlled 24-node
transport graph learns a reusable local operator rather than one fixed
network.

The data generator and training policy follow the
[controlled transport forecast](transport-forecast.md). After training, the
weights are frozen and evaluated without retraining on:

```text
32 and 48 nodes
      x
dense fields and four localized pulses
      x
16 recursive forecast steps
```

`DiffusionGRU` receives the true, degree-preserving permuted, and self-only
topologies for every scope. The node-local LSTM and persistence floor receive
no graph. A state digest before and after all evaluations verifies that the
trained parameters do not change.

## Protocol

Revision `8bcce51d853590184996db4d9d54a571cc993052` produced the six
DiffusionGRU runs. Revision
`ca98d4a9d57311d31398b4f0c8ec3bc6708eabe5` produced the three matched
LSTM runs after consolidating its four evaluation scopes under one frozen
model per seed.

Both models train on 128 eight-step trajectories from host-random seed
`20260729` and select a checkpoint on 32 trajectories from seed `20260730`.
Each training field yields four causal four-snapshot windows and next-step
targets. Each transfer scope contains eight independent 20-step trajectories.
Dense and pulse transfer fields use seeds `20260861` and `20260862` at 32
nodes, then `20260877` and `20260878` at 48 nodes.

The matched settings are hidden width 8, batch size 64, Adam at `0.01`, 30
epochs, validation-RMSE checkpoint selection, and model seeds 0, 1, and 2.
DiffusionGRU has 681 parameters; the node-local LSTM has 361.

```console
uv run --locked python -m experiments.run transport_transfer DEV=CPU MODEL=diffusion_gru NODES=all INITIAL=dense SEED=0 EPOCHS=30 HISTORY=4 HORIZON=16 BS=64 HIDDEN=8 LR=0.01
```

Repeat with `INITIAL=pulse` and model seeds 1 and 2. The LSTM control evaluates
only the two transfer sizes and trains once per model seed:

```console
uv run --locked python -m experiments.run transport_transfer DEV=CPU MODEL=lstm NODES=unseen INITIAL=both SEED=0 EPOCHS=30 HISTORY=4 HORIZON=16 BS=64 HIDDEN=8 LR=0.01
```

Repeat with model seeds 1 and 2. Every recorded run completed within the
runner's unchanged 600-second limit. One exploratory LSTM process that also
evaluated the 24-node training size hit that limit under local contention and
produced no observation. The frozen protocol omits that irrelevant source-size
scope and evaluates both transfer sizes from one training run.

## Results

The tables report mean RMSE plus sample standard deviation across three model
seeds. Persistence is fixed because the transfer fields are fixed.

### One step

| Nodes | Initial field | True graph | Permuted graph | Self only | LSTM | Persistence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | dense | **0.002974 ± 0.000934** | 0.057599 ± 0.009753 | 0.038316 ± 0.008562 | 0.019723 ± 0.002799 | 0.023087 |
| 32 | pulse | **0.001671 ± 0.000867** | 0.022464 ± 0.003940 | 0.017412 ± 0.004042 | 0.007535 ± 0.000686 | 0.009984 |
| 48 | dense | **0.002378 ± 0.000839** | 0.048216 ± 0.008330 | 0.037299 ± 0.008545 | 0.017915 ± 0.002587 | 0.021852 |
| 48 | pulse | **0.001686 ± 0.000899** | 0.019469 ± 0.003312 | 0.014806 ± 0.003418 | 0.006874 ± 0.000735 | 0.008460 |

### Sixteen-step rollout

| Nodes | Initial field | True graph | Permuted graph | Self only | LSTM | Persistence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | dense | **0.111566 ± 0.001623** | 0.234199 ± 0.003159 | 0.186945 ± 0.007927 | 0.145215 ± 0.000985 | 0.187691 |
| 32 | pulse | **0.041033 ± 0.000186** | 0.080628 ± 0.001387 | 0.090401 ± 0.004037 | 0.051327 ± 0.000880 | 0.076943 |
| 48 | dense | **0.085920 ± 0.001033** | 0.175722 ± 0.002407 | 0.173907 ± 0.009923 | 0.123477 ± 0.000689 | 0.168571 |
| 48 | pulse | **0.036867 ± 0.000220** | 0.073616 ± 0.001385 | 0.076986 ± 0.003097 | 0.050715 ± 0.000905 | 0.066419 |

### Step sixteen

| Nodes | Initial field | True graph | Permuted graph | Self only | LSTM | Persistence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | dense | **0.009855 ± 0.004262** | 0.228502 ± 0.000830 | 0.245218 ± 0.006491 | 0.115754 ± 0.000683 | 0.254992 |
| 32 | pulse | **0.005969 ± 0.001404** | 0.076587 ± 0.001069 | 0.110273 ± 0.002840 | 0.040085 ± 0.001759 | 0.097847 |
| 48 | dense | **0.007929 ± 0.002542** | 0.179053 ± 0.000910 | 0.210685 ± 0.007671 | 0.103811 ± 0.000329 | 0.212729 |
| 48 | pulse | **0.006184 ± 0.001329** | 0.071969 ± 0.001117 | 0.095722 ± 0.002225 | 0.042069 ± 0.001585 | 0.086018 |

True-topology DiffusionGRU wins every paired comparison against the permuted
graph, self-only graph, LSTM, and persistence in all three seeds. Against the
strongest alternative in each scope, its mean rollout-RMSE reduction ranges
from 20.1% to 30.4%. The 32- and 48-node scopes agree for both dense and
localized initial fields.

## Decision

The transfer claim required true-topology DiffusionGRU to beat its permuted and
self-only controls plus the node-local LSTM on one-step and 16-step rollout
RMSE in at least two of three seeds. Both unseen graph sizes had to agree; the
24-node training size could not pass the gate.

The gate passes. One frozen set of local diffusion and recurrent parameters
uses the correct sparse topology on graph sizes never seen during training.
The positive result is not explained by node count, smooth dense fields,
persistence, or a node-local recurrent model.

No new public API is promoted. `Graph`, `DirectedDiffusion`, and
`DiffusionGRU` already own the reusable contracts and now have stronger
evidence. The synthetic generator, pulse policy, graph-family construction,
scope selectors, state digest, recursive evaluation, and training loop remain
research-only. Sequential evaluation is sufficient, so the experiment does
not justify batching different graphs.

## Limits

The source and target graphs come from one synthetic graph family and share
one deterministic, noiseless, fully observed transport law. The generator and
`DirectedDiffusion` express the same known local operator independently. Node
semantics, coefficients, feature count, sampling interval, and topology type
do not change. The LSTM is a graph-free control, not a parameter-matched
architecture ablation.

This proves controlled size transfer, not transfer across cities, physical
systems, or process regimes. More experiments on this generator would mostly
measure an already identified operator. The next model-quality stage should
retain these controls on a real physical sensor network with missingness,
exogenous context, and measured topology.
