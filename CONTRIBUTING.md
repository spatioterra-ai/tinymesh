# Contributing to tinymesh

tinymesh accepts changes that make sparse spatiotemporal learning simpler,
more correct, or more measurable.

This guide adapts Tinygrad's
[contribution policy](https://github.com/tinygrad/tinygrad/blob/2864036e8ef99cab0a14d3bc19574a052d09b1b8/README.md#contributing)
to tinymesh. We keep its small-change, regression-test, benchmark, and
complexity discipline. We differ where the projects differ: documentation is
part of this early research project, and transparent AI assistance is allowed.

Read [README.md](README.md), the relevant document under [`docs/`](docs/), and
[voice.md](voice.md) before contributing.

## The bar

- One change has one purpose.
- Prefer an existing primitive over a new abstraction.
- Reduce concepts, not newlines. Dense code is not simple code.
- Fix root causes. Do not hide a backend limitation behind a compatibility
  layer or special case.
- Make behavior deterministic and contracts explicit.
- Test behavior changes. Add a regression test for every bug fix.
- Benchmark every performance or scaling claim on named hardware.
- Keep topology sparse. A path called sparse must not materialize node-pair or
  node-edge state at network scale.
- Do not mix dependency pins, refactors, features, generated files, or
  whitespace unless they are inseparable.

## Good contributions

- A small bug fix with a focused regression test.
- A simplification that removes a concept or unifies duplicated paths.
- A reusable mesh operation required by more than one current model composition.
- A reproducible experiment that resolves an architecture decision.
- A non-brittle correctness, gradient, equivariance, or scaling test.
- Documentation that makes an implemented contract, limitation, or result more
  precise.
- Dead-code removal.

Speculative model-zoo entries, general GIS infrastructure, trainer frameworks,
PyTorch compatibility layers, and abstractions without a current caller are out
of scope.

## Before changing code

1. Name the current caller, workload, or documented contract and its concrete
   correctness, scaling, or metadata consequence. A reachable synthetic state
   alone is not a demonstrated product defect.
2. Reproduce the problem through that caller on current `main`.
3. Record the revision, command, expected result, and observed result. Include
   the dependency revision, device, and shapes when they affect the claim.
4. Trace only adjacent paths that can change or consume the claimed fact: for
   example, backward for a gradient claim or optimization search for compiler
   metadata.
5. Identify the smallest owner of the fix. tinymesh owns mesh contracts and
   compositions; tinygrad owns primitive tensor and compiler behavior. A
   scaling observation alone does not decide which repository must change.
6. Check whether a clean prerequisite refactor can make the behavior change
   small and obvious. Keep that refactor independently useful.
7. If no live caller or documented contract depends on the behavior, stop. Do
   not turn a latent hypothesis into product code or an upstream issue.
8. Open an issue first when the API, ownership, sparse invariant, or dependency
   direction is unresolved. Small proven fixes can go directly to a pull
   request.

## Pull requests

Use an imperative, concrete title such as `Prove sparse aggregation
feasibility`. Keep the body proportional to the change:

- why the change should exist;
- what contract or behavior changes;
- the evidence that proves it;
- any important limitation or deliberate exclusion.

A tiny PR may need only two sentences and one test command. A performance PR
needs before-and-after measurements, shapes, device, revision, and enough setup
to reproduce them. Do not narrate every changed file or paste routine tool logs.

Reviewers should be able to answer three questions from the diff and body:

1. Why is this the right owner?
2. Why is this the smallest complete change?
3. What fresh evidence proves the claim?

## Verification

Run focused checks first, then the full suite when the change is broad:

```console
uv sync --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Run `uv run python experiments/sparse_aggregation.py` when changing sparse
aggregation, lowering assumptions, or scaling evidence. Test every backend
named in the claim. A passing numerical result does not prove sparse complexity;
inspect operation growth and intermediate shapes too.

Run `uv run python experiments/csr_aggregation.py` when changing the CSR
candidate, its transpose backward, or degree-skew evidence.

Dependency changes must keep `pyproject.toml`, `uv.lock`, and the corresponding
reference submodule on the same exact revision. Change a gitlink in a dedicated
pull request and record the upstream delta and compatibility evidence.

## AI assistance

AI is a tool, not a substitute for ownership. Disclose material AI assistance
in the pull request. The contributor must understand every changed line, rerun
every reported check, verify every source, and answer review questions directly.

Generated bulk, invented evidence, hidden AI authorship, or tests weakened to
make a change pass will be closed. Do not imitate another contributor's voice.
Use [voice.md](voice.md) as the project standard.
