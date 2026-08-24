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
order for each view, shuffling ties with the configured seed before stable sorting. A
separately shuffled order supplies the deterministic random baseline. Every ten
percent it reports all components and the largest component over the stable
1,899-node universe, including isolates.

## Observation

The revision-bound observation will be recorded after the executable contract
passes from a clean commit.

## Decision

The projection, strength policies, overlap, and connectivity measurements
remain research-only. `TemporalEdges` and `college_msg()` already own the
reusable event and source facts; no `Graph` method, automatic snapshot,
community API, or analytics dependency follows from this experiment.
