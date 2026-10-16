# D12 source-reported feature status view

The policy is to preserve useful factual evaluation data from failed D12
executions without teaching failed assistant actions or inventing a cause or
repair. The complete failed conversation is context. The only positive target
is the exact structured status already reported by the source evaluator.

1. Read the original D12 cooperation trajectory and its bound `eval.json`.
2. Require a failed source outcome and a well-formed two-feature evaluation
   whose boolean `passed` fields agree with `both_passed == false`.
3. Keep the complete observed transcript in a tool/context message. Do not
   promote any assistant message from that failed transcript.
4. Emit only the exact feature identifiers and pass booleans, plus structured
   `apply_status`, merge status and integer feature exit codes when present.
   Do not copy error prose or claim a cause, fix, local execution or oracle
   verification.
5. Preserve source/split/protection provenance and create a view-qualified
   identity so this derived status task cannot collide with the original
   failed trajectory.

The source annotation and safe derivation are natural-language policy. Exact
source joining, schema checks, transcript serialization, target construction,
hashes and split/protection verification are crisp shared adapter plumbing.
