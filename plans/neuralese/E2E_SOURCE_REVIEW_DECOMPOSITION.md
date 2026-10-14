Filter a mined training cohort using the shared source-review registry.

1. Read the immutable cohort's chain membership and exact dataset:item source groups. Obtain pending unpinned groups from the existing source-review matcher; record its registry SHA.
2. When a train record belongs to a pending source, omit its whole trajectory. Other records in that trajectory depend on the rejected context. Keep independently useful trajectories, including trajectories that merely share a seed paragraph. Preserve held chains and record their quality flags separately.
3. Copy kept record bytes and keep every edge only when both endpoints remain in its original chain. Verify unique membership, unchanged splits, endpoint closure, and per-chain edge counts. Keep the shared immutable piece table.
4. Write an explicit omission for every rejected record with its trajectory and matching source group. Publish a new immutable cohort and its hashes; do not rewrite the source cohort or infer a model/DPO negative.

The source judgments are natural-language review entries; the existing matcher implements exact identity lookup. Crisp code is justified for streaming bytes, graph integrity checks, hashes and atomic publication. The same review policy also applies during future assembly, using whole trajectories rather than truncating their context.
