# pr-shepherd

Tier 1 Flagship A1: automated pull-request review combining crisp routine-hunk skipping, calibrated finite-record
classification, grounded quote verification, debiased finding ranking, asymmetric risk escalation, and verified
documentation-fix transactions.

```sh
natlang check applications/pr-shepherd
```

## Architecture

- **Crisp fast-path (`pluggable`)**: `createTrivialFilter` skips lockfiles, snapshots, generated artifacts,
  whitespace-only diffs, and import-order-only diffs in pure TypeScript (0 model calls; `shadow` mode tracks agreement
  against `isTrivialHunk.nl`).
- **Finite-record `readout: decision`**: `classifyHunk.nl` scores `{ kind, risk }` jointly across 15 finite values.
- **Zero hallucinated quotes (`ensures:`)**: `checkHunk.nl` enforces `quotesMatch(result, hunk)` via
  `checkHunk/quotesMatch.ts` plus a semantic postcondition against `[hunk]`.
- **Untrusted PR diffs**: `Hunk.patch` is `Untrusted<string>`, fencing adversarial prompt-injection comments as data.
- **Asymmetric safety gate (`escalate`)**: `mergeVerdict.nl` escalates to `teacher` when confidence `< 0.85` **or**
  `P(block) >= 0.25`.
- **Verified doc fixes (`proposeAndVerify`)**: `proposeDocFix.nl` runs inside a private `Folder` transaction and only
  commits when the exact verifier accepts the proposal.
