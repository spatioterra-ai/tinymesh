# JEPA mechanics

This stage asks one question: can tinygrad and tinymesh express the asymmetric
learning mechanism behind I-JEPA and Graph-JEPA without a new framework or
dependency?

## Protocol

```text
context graph patch -- online SAGE encoder -- predictor(position) -- prediction

target graph patches -- EMA SAGE encoder -- stop gradient ------------ target
                                                                   |
                                                              latent MSE
```

Sixteen deterministic examples encode two latent values in one context patch
and two distinct target patches. Every patch is the same three-node path. The
online encoder and predictor receive gradients; the target encoder begins as an
exact copy and moves only through an exponential moving average of the online
encoder.

The stage has three failure controls:

- reversed examples test whether predictions match their own targets;
- zero position tokens test whether the two targets are distinguishable;
- target latent variation across examples tests the simplest form of collapse.

The protocol is deliberately synthetic. It isolates mechanics before data,
partitioning, random-walk position, hyperbolic projection, or downstream
representation quality can confound the result.

## Decision

At tinymesh revision
[`0b1b9a5`](https://github.com/spatioterra-ai/tinymesh/tree/0b1b9a51d7ecf75981bfc2df1facb3c9b8dfac3b),
the asymmetric mechanism trains on CPU and Metal with the pinned tinygrad
revision `1095bbe`. The two devices agree within `1e-7` on the reported losses.

| Measurement | CPU | Metal |
| --- | ---: | ---: |
| Initial latent MSE | 0.319204 | 0.319204 |
| Aligned latent MSE | 0.029169 | 0.029169 |
| Reversed-target MSE | 0.061945 | 0.061945 |
| Zero-position MSE | 0.096477 | 0.096477 |
| Target variation across examples | 0.171856 | 0.171856 |
| Target gradient | 0 | 0 |

Aligned loss fell by `90.9%`. Reversing examples made it `2.12x` worse and
removing target position made it `3.31x` worse, so the predictor uses both
sample content and target identity. The EMA target moved by `5.1267` in summed
absolute parameter distance while receiving no gradient.

This proves the learning mechanism and its tinygrad execution, not that graph
topology helps, that the representation transfers, or that the three-node
fixture captures Graph-JEPA. `PatchEncoder`, `Predictor`, EMA, and the task stay
research-only. The existing `Graph` and `SAGEConv` APIs already own all reusable
math used here.

## Sources

[I-JEPA v3](https://arxiv.org/abs/2301.08243v3) supplies the asymmetric online
and EMA target encoders, stop-gradient target, position-conditioned predictor,
and latent L2 objective. [Graph-JEPA
v3](https://arxiv.org/abs/2309.16014v3) motivates graph patches and graph
encoders; its partitioning, random-walk positional encoding, smooth-L1 loss,
and hyperbolic projection are outside this stage.

## Reproduce

```console
uv run --locked python -m experiments.run jepa_mechanics DEV=CPU EMA=0.99 HIDDEN=8 LR=0.01 SAMPLES=16 SEED=0 STEPS=80
uv run --locked python -m experiments.run jepa_mechanics DEV=METAL EMA=0.99 HIDDEN=8 LR=0.01 SAMPLES=16 SEED=0 STEPS=80
```
