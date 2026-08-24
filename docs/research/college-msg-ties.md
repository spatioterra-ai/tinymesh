# CollegeMsg tie structure

CollegeMsg can test whether observable communication intensity aligns with
local overlap and global connectivity without treating messages as friendship.

## Measurement contract

```text
directed timestamped messages                     source truth
              |
              v
final simple undirected contact graph              explicit projection
              |
       +------+------+----------------+
       |             |                |
       v             v                v
message count    active days      reciprocity       separate strength views
       |             |                |
       +------+------+----------------+
              |
       +------+----------------+
       |                       |
       v                       v
neighborhood overlap      edge-removal curves       descriptive measurements
```

Self-messages do not create contacts. For every contacted pair, message count
includes all non-self messages, active days counts distinct UTC Unix days, and
reciprocity records whether at least one message occurred in each direction.
None is called a friendship or combined into an invented strength score.

Neighborhood overlap is the shared-neighbor count divided by the union of the
endpoints' neighbors after excluding the endpoints. An isolated dyad has no
defined denominator and remains explicitly unavailable. Embeddedness is the
shared-neighbor count; zero identifies a local bridge in the final simple graph.

Message count and active days use fixed power-of-two bins. Reciprocity remains
a two-category view. The experiment removes edges in ascending and descending
order for each view, shuffling ties with the configured seed before stable
sorting. A separately shuffled order supplies the deterministic random
baseline. Every ten percent it reports all components and the largest component
over the stable 1,899-node universe, including isolates.

## Observation

Revision
[`b41c318`](https://github.com/spatioterra-ai/tinymesh/tree/b41c318251bae3f55775ca29bc28d932c4797643)
produced this CPU witness in 1.347 seconds:

```console
uv run --locked python -m experiments.run college_msg_ties SEED=0
```

The 59,835 messages contain no self-message, project to 13,838 contacted pairs,
and include 6,458 reciprocal pairs. The final graph has four components; its
largest contains 1,893 of 1,899 nodes. Of all contacts, 3,969 are local bridges
and three isolated-dyad overlaps are unavailable.

Message count aligns monotonically with greater mean overlap across every
observed bin:

| Messages | Ties | Mean overlap | Mean embeddedness | Local bridges |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5,231 | 0.023454 | 2.465 | 1,971 |
| 2-3 | 4,322 | 0.029111 | 2.988 | 1,194 |
| 4-7 | 2,487 | 0.033842 | 3.645 | 548 |
| 8-15 | 1,182 | 0.034308 | 4.133 | 181 |
| 16-31 | 428 | 0.041305 | 5.250 | 49 |
| 32-63 | 145 | 0.041455 | 4.834 | 23 |
| 64-127 | 37 | 0.042778 | 5.378 | 3 |
| 128-255 | 6 | 0.043325 | 7.833 | 0 |

The active-day view rises from `0.026577` mean overlap for 8,725 one-day ties
to `0.040628` for 236 ties active on 8-15 days. The two higher bins contain only
35 and 7 ties and are not monotone, so they do not support a stronger trend
claim. Reciprocal contacts have `0.032750` mean overlap and `3.399` mean
embeddedness, versus `0.025375` and `2.846` for one-way contacts. Local bridges
are 23.5% of reciprocal contacts and 33.2% of one-way contacts.

Weak-first removal consistently reduces the giant component more than
strong-first removal. The random column is one common seed-zero baseline; the
mean is the arithmetic mean of largest-component fractions at the eleven
ten-percent checkpoints, including zero and complete removal.

| Removal measure | Weak-first mean | Strong-first mean | Random mean | Weak-first largest at 50% | Strong-first largest at 50% | Random largest at 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Message count | 0.6369 | 0.8012 | 0.7447 | 1,292 | 1,785 | 1,611 |
| Active days | 0.6880 | 0.7803 | 0.7447 | 1,476 | 1,716 | 1,611 |
| Reciprocity | 0.6768 | 0.7816 | 0.7447 | 1,354 | 1,749 | 1,611 |

For reciprocity, weak-first removes one-way contacts before reciprocal contacts;
strong-first reverses that order. These curves support the descriptive weak-tie
pattern in this projection. They do not estimate what would happen if social
relationships were removed: activity, degree, homophily, observation coverage,
and within-bin ordering remain confounded, and the final graph uses the complete
observation period.

## Decision

The projection, strength policies, overlap, and connectivity measurements
remain research-only. `TemporalEdges` and `college_msg()` already own the
reusable event and source facts; no `Graph` method, automatic snapshot,
community API, or analytics dependency follows from this experiment.

The evidence closes this final-graph stage. A temporal bridge-lifecycle study
would require a new contract over strict prefixes; it should not reuse these
complete-period measurements as if they were available earlier.
