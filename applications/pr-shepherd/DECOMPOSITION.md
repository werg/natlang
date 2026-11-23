# PR Shepherd (`applications/pr-shepherd/`): Decomposition

Tier 1 Flagship A1 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3 & §3.5): automated pull-request review with
crisp routine-hunk specialization, verbatim quote grounding, debiased finding ranking, asymmetric block escalation,
and verified documentation-fix transactions.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `shepherd.ts` & `checkHunk/quotesMatch.ts`):**
  - `isTrivialHunkCrisp`: deterministic detection of generated files, lockfiles, snapshots, whitespace-only diffs, and
    import-order-only diffs, wrapped in `pluggable` (`crisp | nl | shadow`) to serve >= 60% of routine hunks with zero
    model calls.
  - `quotesMatch`: protected exact postcondition helper (`checkHunk/quotesMatch.ts`) guaranteeing every finding's
    `quote` appears verbatim in `hunk.patch` (0 hallucinated quotes).
  - Fan-out over `(non-trivial hunks × applicable rules)` via `semanticMap` with bounded concurrency.
  - Debiased pairwise Copeland ranking via `semanticRank`.
  - Asymmetric risk-gated escalation via `escalate(mergeVerdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } })`.
  - Transactional documentation fix installation via `proposeAndVerify` over a `Folder` with exact verifier
    `defaultVerifyDocFix`.

- **Typed semantic policy (`.nl` leaves):**
  - `classifyHunk.nl` (`readout: decision`, returns finite record `HunkClass = { kind, risk }`): classifies each
    non-trivial hunk across 15 finite `(kind, risk)` outcomes.
  - `isTrivialHunk.nl` (`readout: decision`, returns `boolean`): semantic counterpart for `pluggable` shadow validation.
  - `checkHunk.nl` (`ensures:` `quotesMatch(result, hunk)` + judged predicate against `[hunk]`): checks one hunk
    against one rule bound via `withContext`.
  - `mergeVerdict.nl` (`readout: decision`, returns `'approve' | 'request-changes' | 'block'`): calibrated merge gate.
  - `proposeDocFix.nl`: drafts a doc update addressing grounded findings and verifier feedback.

## Safety Boundaries

- `Hunk.patch` is typed `Untrusted<string>`: PR code and adversarial comments are fenced as outside data by the runtime
  and cannot hijack instructions.
- `checkHunk/quotesMatch.ts` is auto-protected from edits by `ensures:` and obeys finite-iteration policy (no `while`
  loops).
- `proposeAndVerify` isolates doc edits in a private `Folder` copy and installs changes only when the exact verifier
  returns zero problems.
