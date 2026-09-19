# P02 — Semantic replication / notional CRDT

Status: proposed implementation. [Shared capabilities](README.md). This is intentionally semantic merging performed by natlang, not a crisp CRDT merge algebra.

## Natlang prerequisites

C0 already expresses both merge strategies. C2 is the essential API change: explicit shared root seed, model configuration and logical invocation seed derivation. C4 records which inputs/context actually produced a result. No new merge tool, type or convergence primitive is required. C5 later delivers replica updates; independent documents may eventually use C6.

## Programme and typed boundary

Build two entry points: `merge_history.nl(base, updates, policy) -> MergeResult` and `apply_update.nl(state, update, base, policy) -> MergeResult`. Helpers identify intent, compare overlapping changes, merge meaning, preserve unresolved alternatives and explain decisions. Use finite Fold when comparing sequential history reduction; also test whole-history and grouped reduction variants.

Represent document content and updates as ordinary typed records with stable update IDs, causal parents, base revision and author-provided content. `MergeResult` contains merged content, source references and unresolved ambiguities. No native object needs to cross the type boundary.

## Crisp environment and randomness

Only exact preparation is crisp: deduplicate transport deliveries, select a documented presentation order, calculate hashes and compare results. It never decides what an edit means or which meaning wins. Model/seed settings are host configuration, not arguments the weak interpreter must choose.

Pin source, model/tokenizer, decoding profile and context formatting. Derive call seeds from the same logical merge identity, not replica-local IDs or arrival-thread order. Treat external retrieval as an explicit frozen input if used. The initial implementation needs no shared mutable eval environment or network service; two independently instantiated runs are enough.

## Reduction and stream shape

History mode recomputes from an agreed base and update set. State/update mode folds incoming changes into current state. Record order dependence rather than presenting equal seeds as proof of commutativity. At synchronisation points, compare recomputation from a common history with incremental results. Semantic history compression is another natlang function whose effect on preservation must be measured.

Live transport later supplies `Fold<ReplicaEvent, ReplicaState>`. Separate transport arrival from the merge-input presentation policy. Late updates can invalidate a prior local result; retain alternatives until the selected synchronisation policy resolves them. A model-profile mismatch is an explicit unsupported replication configuration.

## Delivery and checks

1. Run identical finite histories twice under the same profile. Gate: repeatability measured separately from quality; investigate any equal-input output mismatch.
2. Compare history and state/update strategies on disjoint, overlapping, contradictory, delete/edit and rename examples. Gate: source updates remain available; semantic judgments are independently graded.
3. Add two replica streams, disconnect/reconnect and optional agreed checkpoints. Gate: disagreement is visible, not silently replaced by a claimed convergence guarantee.

Use permutation/grouping experiments, duplicates and summarisation stress cases. No test requires the merge itself to satisfy a conventional crisp algebra.

## Trace, teacher and portability

Record full input presentation, source/profile hashes, derived seeds, intermediate semantic reductions and final digest. Teacher references accept multiple justified meanings where appropriate; replica agreement alone is not a quality oracle. A new backend must demonstrate the required repeatability before joining this execution profile, while remaining free to run unrelated natlang programmes without that guarantee.
