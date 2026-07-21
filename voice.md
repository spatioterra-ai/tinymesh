# Tinymesh voice

tinymesh sounds calm, direct, exact, and evidence-first.

The voice borrows useful habits from Tinygrad and George Hotz's technical
writing: start from the real problem, use concrete examples, make claims
falsifiable, prefer a complete simple formulation, and admit uncertainty or a
changed mind. It does not copy a person's persona. Profanity, insults,
provocation, hype, and maintainer-only terseness do not belong in project
communication.

## Core rules

1. Lead with the result, decision, blocker, or exact question.
2. State observed facts before interpretation.
3. Make claims falsifiable with revisions, shapes, devices, commands, numbers,
   or counterexamples.
4. Use the fewest words that preserve the reasoning.
5. Name uncertainty plainly: `unknown`, `not tested`, `blocked`, or `inferred`.
6. Keep one issue, pull request, and comment thread about one concern.
7. Critique artifacts and contracts, never the person who produced them.
8. Prefer a direct noun and verb over adjectives, slogans, or framework jargon.
9. Do not perform confidence. Strong language requires strong evidence.
10. Stop when the reader has what they need to decide or act.

## General prose

Use active voice, short paragraphs, and sentence-case headings. Contractions are
fine. Lists are for genuinely parallel facts; prose is better for one causal
argument. Put exact identifiers, commands, shapes, and errors in code formatting.

Use names consistently: `tinymesh`, `tinygrad`, `PyTorch`, and `PyG`.

Words that carry an evidence obligation:

| Claim | Required evidence |
| --- | --- |
| `correct` | Expected behavior and a focused test |
| `sparse` | Data structure plus asymptotic and measured work growth |
| `faster` | Before and after on named hardware with the same workload |
| `simpler` | The concept, branch, owner, or code path removed |
| `supports` | A stable contract and representative tests |
| `equivalent` | Parity over the relevant outputs and internal invariants |

Avoid `obviously`, `just`, `easy`, `robust`, `clean`, `optimal`, and
`production-ready` unless the sentence proves the term. Never call a change
“massive,” “revolutionary,” or “game-changing.”

## Pull requests

The title is an imperative outcome, normally under 72 characters and without a
period.

Good:

- `Prove sparse aggregation feasibility`
- `Keep tinygrad pins on one revision`
- `Reject dense message-passing lowering`

Bad:

- `Major graph improvements`
- `Refactor stuff`
- `WIP: awesome new architecture`

Start the body with why the change should exist, not a file inventory. Then give
the smallest evidence needed to evaluate it. Use headings only when the body is
long enough to need them.

Small change:

```markdown
`Mesh.validate` accepted destinations outside the node range. Rejecting them at
construction keeps invalid topology out of every downstream operation.

Verified with `uv run python -m unittest tests.test_mesh`.
```

Measured change:

```markdown
Sparse aggregation currently grows with node-edge pairs because it constructs
an `[N, E, H]` mask. This change lowers destination sum without that axis.

On Metal at tinygrad `<sha>`, doubling `N` and `E` changes operations from
`<before>` to `<after>`. Forward, repeated-destination backward, isolated-node,
and permutation tests pass with `<command>`.
```

Do not dump routine logs, praise the implementation, repeat the title, or list
every file changed. State exclusions when they prevent overclaiming.

## Issues

An issue is a reproducible problem or a bounded design decision, not a roadmap
pitch.

A bug report contains:

- exact tinymesh and tinygrad revisions;
- device and relevant environment;
- smallest copyable reproducer;
- expected and observed behavior;
- full exact error or measurement;
- why the result violates a documented contract.

A proposal contains:

- the missing capability in user terms;
- the current path and its measured cost;
- the smallest proposed contract;
- the unresolved decision.

Good titles:

- `Sparse aggregation scales with node-edge pairs`
- `Repeated destinations produce the wrong Metal gradient`
- `Decide ownership of coordinate reference metadata`

Bad titles:

- `Graphs are broken`
- `We need a better GNN system`
- `Idea: build the ultimate mesh framework`

Do not bundle gather, segmented reduction, temporal recurrence, and a model zoo
into one issue because one application needs all four. Prove and discuss one
primitive at a time.

## Comments and reviews

Lead with the finding and its consequence. Prefix a comment when the review
role is otherwise ambiguous:

- `blocking:` correctness, privacy, sparse complexity, or contract violation;
- `question:` missing evidence or unclear intent;
- `suggestion:` a non-required improvement;
- `nit:` a truly optional local polish point.

Good:

```text
blocking: this creates an [N, E, H] mask, so doubling N and E quadruples work.
Can the reduction stay proportional to E * H?
```

```text
question: what invariant does MeshManager own that Mesh does not?
```

Bad:

```text
This is crazy and messy.
```

```text
Looks good!
```

Explain a rejection with the violated contract or missing evidence. Answer the
question asked before adding context. Acknowledge a correction directly:
`You're right; this benchmark changed two variables. I reran it with only N
changing.` Resolve a thread when the finding is addressed, not when the author
merely replies.

## Commits and releases

Commit subjects use the same imperative style as pull-request titles and cover
one purpose. Release notes say what users can now rely on, what changed, and
what remains experimental. Never turn a passing toy example into a platform or
production claim.

## AI-written material

AI-assisted prose follows the same bar and is disclosed under
[CONTRIBUTING.md](CONTRIBUTING.md). Remove generic summaries, ornamental
headings, fake quotations, repetitive conclusions, and claims that no person
has verified. Do not ask a model to make generated text “sound human” or to
imitate a named person.

## Sources and limits

Tinygrad's current
[contribution guide](https://github.com/tinygrad/tinygrad/blob/2864036e8ef99cab0a14d3bc19574a052d09b1b8/README.md#contributing)
supplies the small-diff, readability, benchmark, regression-test, API-parity,
and replay discipline. Representative accepted changes show unification
([`#9114`](https://github.com/tinygrad/tinygrad/pull/9114)), exact performance
evidence ([`#14400`](https://github.com/tinygrad/tinygrad/pull/14400)), and tests
whose parity can be inspected ([`#8947`](https://github.com/tinygrad/tinygrad/pull/8947)).

The blog posts
[“Can tinygrad win?”](https://geohot.github.io/blog/jekyll/update/2025/07/06/can-tinygrad-win.html),
[“Five years of tinygrad”](https://geohot.github.io/blog/jekyll/update/2025/12/29/five-years-of-tinygrad.html),
[“AI Coding”](https://geohot.github.io/blog/jekyll/update/2025/09/12/ai-coding.html),
[“The Eternal Sloptember”](https://geohot.github.io/blog/jekyll/update/2026/05/24/the-eternal-sloptember.html),
and
[“I love LLMs, I hate hype”](https://geohot.github.io/blog/jekyll/update/2026/07/12/i-love-llms.html)
support the emphasis on first-principles formulation, measurable goals, direct
examples, tool skepticism, human error correction, and willingness to revise a
claim. They are personal essays, not contribution rules; their confrontational
style is deliberately excluded here.
