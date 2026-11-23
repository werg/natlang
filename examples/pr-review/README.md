# pr-review: automated pull-request review with grounded findings and asymmetric escalation

Automates repeat code-review work on a small local model using `@natlang/node/std` and verified contracts. This is the
small teaching version; `applications/pr-shepherd` is the full application (crisp routine-hunk fast path, hunk
classification, verified documentation fixes) built from the same functions.

```sh
natlang check examples/pr-review
natlang run examples/pr-review
natlang test examples/pr-review
```

- **Untrusted diffs**: `Hunk.patch` is `Untrusted<string>` (including a prompt-injection comment in `diff.json` that the
  runtime fences as data and never follows).
- **Zero hallucinated quotes (`ensures:`)**: `check_hunk.nl` declares `check: "quotesMatch(result, hunk)"`, backed by a
  protected TypeScript helper (`check_hunk/quotesMatch.ts`) that rejects any finding whose `quote` does not appear
  verbatim in `hunk.patch`.
- **Per-rule specialization (`withContext`)**: binds `check_hunk.nl` once per rule in `rules.json` while keeping its
  `quotesMatch` verifier in context, then maps across `(hunks x rules)` via `semanticMap`.
- **Asymmetric risk escalation (`escalate`)**: `merge_verdict.nl` (`readout: decision`) runs locally, and
  `escalate(mergeVerdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } })` escalates whenever confidence is below
  0.85 **or** `P(block) >= 0.25`.
- **Debiased ranking (`semanticRank`)**: orders findings by engineering criticality using symmetric pairwise comparisons.
