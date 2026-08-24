# CollegeMsg temporal closure

Static clustering cannot tell whether a triangle preceded or followed its
third edge. CollegeMsg supplies the missing order, but its edges record
communication rather than friendship.

## Source boundary

The [SNAP CollegeMsg source](https://snap.stanford.edu/data/CollegeMsg.html)
contains directed private-message events from an online social network at the
University of California, Irvine. Each row is `source target unix_timestamp`.
The source page reports no redistribution license, so tinymesh does not commit
the artifact. The experiment accepts only the 345,339-byte gzip with SHA-256
`50ae2d98ed3bad9ddb18dbd495a89e5e10cfb8f7e86932827db29fc41b41f9fa`.

```text
directed timestamped messages              source truth
              |
              v
groups with equal timestamps               no arbitrary within-time order
              |
              v
undirected first contact per pair           explicit measurement projection
              |
       +------+------+
       |             |
       v             v
prior open wedge   prior non-wedge          mutually exclusive risk sets
       |             |
       +------+------+
              v
       formations / pair-time exposure
```

The parser preserves direction, identity, duplicate messages, and timestamps.
Only the closure measurement projects a directed message to an undirected
observed contact. A pair enters the risk set after both users have appeared and
leaves after its first contact. Contacts that introduce either user are
reported separately because the source cannot establish their prior exposure.

At each distinct timestamp, the experiment first measures elapsed exposure and
classifies every new contact against the graph strictly before that timestamp.
It then adds the whole timestamp group. A non-edge is wedge-exposed when its
endpoints have at least one prior common neighbor. The incremental wedge set
costs `O(degree(source) + degree(target))` per new contact and stores no dense
adjacency or node-pair-by-time table.

## Reference boundary

The pinned graph libraries answer adjacent but different questions:

- PyG Temporal's
  [`TwitterTennisDatasetLoader`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/dataset/twitter_tennis.py)
  returns popularity-filtered mention snapshots and a future-mention target.
  That selection changes the degree and closure population.
- PyG's
  [`BitcoinOTC`](https://github.com/pyg-team/pytorch_geometric/blob/5c6461b2305ad068a6d61165b3c55852a11aaa41/torch_geometric/datasets/bitcoin_otc.py)
  bins timestamped signed trust ratings and gives edges a fixed ten-window
  lifetime. Trust, sign, bin width, and edge lifetime are different contracts.
- PyG's
  [`JODIEDataset`](https://github.com/pyg-team/pytorch_geometric/blob/5c6461b2305ad068a6d61165b3c55852a11aaa41/torch_geometric/datasets/jodie.py)
  preserves temporal interactions but offsets destination identities into a
  bipartite user-item graph, where ordinary user-user triadic closure does not
  apply.

Tinymesh therefore uses no framework container or snapshot policy here. The
executable evidence consumes only the checksum-pinned message stream, so the
catalog records no reference gitlink.

## Observation

Revision
[`2100277`](https://github.com/spatioterra-ai/tinymesh/tree/210027777f43f25850dcbe234f630e174c496b34)
produced this witness:

```console
uv run --locked python -m experiments.run college_msg_closure SOURCE=.data/CollegeMsg.txt.gz
```

| Source property | Observation |
| --- | ---: |
| Messages / users | 59,835 / 1,899 |
| Directed message pairs | 20,296 |
| Undirected contacted pairs | 13,838 |
| Repeated messages after first contact | 45,997 |
| First contacts introducing a user | 1,826 |

The remaining 12,012 first contacts have both endpoints in the prior observed
population:

| Prior relation | Formations | Pair-seconds exposed | Formations per million pair-days |
| --- | ---: | ---: | ---: |
| Open wedge | 5,609 | 4,501,578,548,508 | 107.655 |
| Non-wedge | 6,403 | 17,487,499,269,136 | 31.635 |

The wedge incidence rate is 3.403 times the non-wedge rate. This is a temporal
association consistent with triadic closure. It is not a causal estimate:
homophily, user activity, exposure outside the platform, left censoring, and
the undirected projection can all explain part of the difference. A message is
also not evidence of friendship.

## Decision

Retain the source boundary and exact closure measurement as research. Promote
nothing into `src/tinymesh`:

```text
experiments.college_msg_closure   source policy + bounded measurement
src/tinymesh                      unchanged
```

The dataset is useful for studying temporal closure, repeated interaction,
embeddedness, and local bridges as the network chapter develops. Any predictive
stage must first freeze controls for activity, degree, recency, temporal splits,
and negative sampling. A reusable temporal-relation carrier needs a second
runtime caller with the same semantics; MBTA's event-as-node mesh does not meet
that condition.
