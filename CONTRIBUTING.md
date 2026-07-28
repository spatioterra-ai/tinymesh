# Contributing to tinymesh

tinymesh accepts changes that make sparse learning simpler, more correct, or
more measurable. The project adapts tinygrad's
[contribution discipline](https://github.com/tinygrad/tinygrad#contributing):
small changes, readable code, regression tests, and measured performance
claims. tinymesh additionally treats revision-bound research records and
transparent AI assistance as useful.

Read [README.md](README.md) and the relevant path from
[docs/index.md](docs/index.md) before changing code.

## The bar

- Keep one change about one outcome.
- Prefer an existing primitive over a new abstraction.
- Reduce concepts, not newlines.
- Fix the root cause; do not hide backend limits behind compatibility layers.
- Keep behavior deterministic and contracts explicit.
- Test behavior changes, bug regressions, and every claimed backend.
- Benchmark performance claims on named hardware with fixed shapes, revisions,
  and setup.
- Keep topology sparse: network-scale paths must not create node-pair or
  node-edge Cartesian products.
- Do not mix dependency pins, behavior, refactors, generated churn, or
  whitespace unless they are inseparable.

Good changes include focused bug fixes, clear simplifications, reproducible
experiments that resolve a decision, non-brittle tests, precise contract
documentation, and dead-code removal.

Speculative model-zoo entries, general GIS infrastructure, trainer frameworks,
PyTorch compatibility layers, and abstractions without a current caller are out
of scope.

## Before changing code

1. Name the current caller or documented contract and its concrete consequence.
2. Reproduce the problem on current `main`.
3. Record the revision, command, expected result, and observed result. Include
   device and shapes when they affect the claim.
4. Trace only adjacent producers and consumers that can change the result.
5. Put the fix with its smallest owner: tinymesh owns mesh semantics and
   compositions; tinygrad owns tensor primitives and compiler behavior.
6. Stop when no live caller or documented contract needs the change.

Open an issue first only when API ownership, sparse invariants, or a dependency
direction remains unresolved. A small proven change can go directly to a pull
request.

## Pull requests

Use an imperative, concrete title. Keep the body proportional to the change and
answer:

1. Why should this exist?
2. Why is this the smallest owner and change?
3. What fresh evidence proves it?

A performance pull request includes before-and-after measurements, shapes,
device, revision, and enough setup to reproduce them. Do not dump routine logs,
praise the implementation, or hide important exclusions.

## Writing

Lead with the result. State observations before interpretation and make claims
falsifiable with revisions, commands, measurements, or counterexamples. Use
active voice, short paragraphs, and direct names. Critique artifacts and
contracts, never people.

Words such as `correct`, `sparse`, `faster`, `simpler`, `supports`, and
`equivalent` require matching evidence. Avoid hype, ornamental headings, generic
summaries, and imitating another contributor's persona.

## Verification

Run focused checks first. The broad local gate is:

```console
uv sync --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Run `uv run python experiments/sparse_aggregation.py` when changing the rejected
native lowering or its scaling evidence. Run
`DEV=<device> uv run python experiments/csr_aggregation.py` on every backend
named by a CSR, gradient, or degree-skew claim. Run
`DEV=<device> uv run python -m experiments.mean_sage` when changing the first
trainable caller. Run
`DEV=<device> uv run python -m experiments.weighted_aggregation` when changing
edge identity, scalar weighting, or edge gradients.
Run `DEV=<device> uv run python -m experiments.gat` when changing endpoint
projection, target normalization, or attention composition.
Run `DEV=<device> uv run python -m experiments.multi_head_gat` when changing
head composition.

Passing numerical values does not prove sparse complexity. Inspect stored
structure, work growth, and intermediate shapes.

Dependency changes keep `pyproject.toml`, `uv.lock`, and the matching reference
submodule on one exact revision. Change a gitlink in its own pull request and
record the upstream delta and compatibility evidence.

tinygrad experiments follow upstream `master`. At the start of an experiment
stage, advance the runtime dependency, lockfile, and tinygrad submodule together
in a dedicated pull request. Move comparative PyG references only for an
intentional, revision-bound study.

## AI assistance

Disclose material AI assistance in the pull request. The contributor remains
responsible for every changed line, source, check, measurement, and review
answer.

Generated bulk, invented evidence, hidden AI authorship, or weakened tests will
be rejected.
