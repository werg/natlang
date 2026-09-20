# Semantic merge: finite application slice

`merge_history.nl` interprets the meaning of an agreed base and finite causal update set.
The crisp `prepare` helper handles duplicate transport deliveries, base identity and a
documented presentation order. `interpret_history.nl` performs the merge. `finish` checks
that its draft accounts for every update exactly once. A rejected draft keeps the original
updates and their proposed text visible; it does not guess a replacement merge.

`apply_update.nl` supplies a second, incremental reduction. It consumes a prior result and
one new update, then makes another semantic decision while preserving prior unresolved
alternatives. Whole-history and incremental results are compared as experiments; neither
path claims a crisp convergence law.

The nine other entry points cover counters, set-like collections, keyed maps, ordered lists,
parent-linked trees, graphs, calendars, permissions and scenes. Each has its own semantic
leaf and a small exact shape validator. `scripts/generate_semantic_merge_cases.py` expands
20 handwritten scenarios to 80 delivery variants in `scenarios/cases.jsonl`. Related variants share a `group`
and stay together in a deterministic `train` or `eval` split. Their rubrics are semantic expectations,
not executable merge oracles; a teacher still has to produce and pass reviewed trajectories.

To regenerate the scenario inputs:

```bash
.venv/bin/python scripts/generate_semantic_merge_cases.py
```

With a pinned teacher server available, collect a small pilot by naming the exact model:

```bash
.venv/bin/python scripts/collect_semantic_merge_teacher.py \
  codebases/semantic_merge/scenarios/cases.jsonl runs/semantic-merge-teacher.jsonl \
  --model-id MODEL-AND-CHECKPOINT --limit 4
```

The collector stores raw turns and a reduction trace for each case. Its mechanical checks
cover typed completion, source-ID coverage, conflict shape and trace reconstruction.
`semantic_review` remains pending until a reviewer checks the case rubric and the actual
merged meaning. No collected run is automatically admitted as a training reference.

This is a notional CRDT experiment, with no algebraic convergence claim. Replicas must pin
the same model, source, profile, context presentation and root seed before repeatability can
be measured. Equal seeds alone do not imply convergence. The incremental entry point
lets us compare state/update reduction against whole-history reduction. The initial
corpus is finite; live replica transport and model repeatability measurements remain open.
