# METR-LA directed diffusion

This experiment asks whether a causal recurrent model can use METR-LA's
directed distance affinity after unit-weight A3T-GCN failed to beat its
self-only control. It tests one fixed sensor mesh; it does not add a general
space-time graph type.

## Factorized mesh

METR-LA supplies a fixed directed spatial domain and an ordered five-minute
time domain:

```text
space G                 time T
sensor u -> sensor v    ... -> t-1 -> t
             \                 /
              joint state (v,t)
```

The conceptual domain is `G x T`, following the
[time-vertex framework](https://arxiv.org/abs/1705.02307). Execution stays
factorized: one lowered `Graph` is reused while `DiffusionGRU` advances hidden
state through 12 snapshots. Tinymesh never constructs product adjacency or a
dense node-pair tensor.

This is a time-varying traffic field on a directed sensor mesh. The DCRNN
affinity is not a complete street graph, and the source does not declare the
speed or distance units needed to invent physical travel times.

## Inputs

Every input fact is available at its snapshot:

```text
[B,12,207,6]
       |
       +-- normalized speed; missing input becomes 0 after normalization
       +-- observed mask
       +-- sin/cos daily phase
       +-- sin/cos weekly phase
```

Daily and weekly phase are continuous across midnight. Speed normalization,
sensor fallback means, and all baseline statistics use observed training rows
only. Targets, masks, target-time splits, raw-speed metrics, and persistence
remain identical to the existing [METR-LA forecast](metr-la-forecast.md).

## Model

For each period, public Tinymesh components construct three aligned channels.
This is the smallest [DCRNN-style](https://arxiv.org/abs/1707.01926) recurrent
test that preserves the source affinity:

```text
local values --------------------------+
forward affinity diffusion ------------+--> DiffusionGRU --> H[t]
reverse affinity diffusion ------------+
H[t-1] --------------------------------+

H[11] --> ReLU --> Linear(32,12) --> persistence + residual
```

`DirectedDiffusion` source-normalizes the original affinity in both graph
directions. `DiffusionGRU` keeps the local channel beside both transported
channels inside its update, reset, and candidate paths. Sequential unrolling
makes period `t` depend on earlier hidden state; unlike A3T-GCN, periods are not
encoded independently from the same zero state.

At hidden width 32 the model has 11,436 parameters. Twelve recurrent steps
perform 48 sparse sums after the fixed diffusion weights are prepared once.
Topology storage remains `O(N + E)` and no `[N,N]` or `[N,E]` carrier enters
the model.

## Controls and decision

The same initialization, parameters, features, split, batches, loss, and
checkpoint policy receive:

```text
true       original directed support + affinity
permuted   one fixed node relabeling + aligned affinity
self       self edges with unit affinity
```

The first run is validation-only. A topology result advances only if true
affinity beats both structural controls on MAE and RMSE in at least two of
three matched seeds. A stronger node-identity baseline is still required
before calling that a general graph advantage; test remains closed until the
architecture, loss, and budget are frozen.

Later structure has an evidence order:

```text
fixed affinity diffusion
        |
        +-- if useful --> node identity control
                            |
                            +-- delayed edges
                                   |
                                   +-- sparse learned residual graph
```

[STID](https://arxiv.org/abs/2208.05233) motivates the identity control.
[PDFormer](https://arxiv.org/abs/2301.07945) motivates propagation delay.
[Graph WaveNet](https://arxiv.org/abs/1906.00121) motivates an adaptive graph,
but Tinymesh would keep it sparse and parallel to physical support. Typed
relations, regional hierarchy, and physical integration from
[MeshGraphNets](https://huggingface.co/papers/2010.03409),
[RIGNO](https://huggingface.co/papers/2501.19205), and
[PhyMPGN](https://huggingface.co/papers/2410.01337) remain general Tinymesh
research, not claims supported by METR-LA.

## Reproduce

Inspect the protocol or run a bounded optimizer smoke:

```console
uv run --locked python -m experiments.run metr_la_diffusion DEV=CPU
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
```

Revision
[`90f53dca`](https://github.com/spatioterra-ai/tinymesh/commit/90f53dca8ce19833771cf0b35a92672f081139c9)
completed the Metal smoke in 63.48 seconds end to end. The optimizer step
processed 512 full-source windows with 11,436 parameters and 48 sparse calls;
its masked normalized MAE was `0.308893`. This establishes causal forward,
backward, Adam, sparse structure, controls, and revision-bound logging. One
step makes no predictive-quality claim; matched validation evidence remains
pending.
