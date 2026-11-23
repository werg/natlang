# Codemod Studio (`applications/codemod-studio/`): Decomposition

Tier 3 flagship (WS6 §3 A9): repository-scale code migrations where deterministic crisp rules handle covered files with zero LLM calls, remaining files are repaired against an exact verifier (`typecheck + tests`) inside `Folder` transactions, and repeated semantic edits are mined into verified crisp rules for subsequent migrations.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `studio.ts`):**
  - Deterministic rule application (`applyCrispRule`, `applyCrispRules`) running first across all files with zero LLM calls on covered or already-clean files.
  - Isolated transactional workspace updates via `Folder.fromFiles` and `proposeAndVerify` from `@natlang/node/std`, ensuring no unverified file edit is ever accepted.
  - Exact rule admission (`verifyMinedRule`): replays every candidate mined rule against already-migrated pairs and clean baseline files through the verifier before admitting it.
  - Coverage telemetry (`crispFiles`, `semanticFiles`, `unchangedFiles`, `failedFiles`, `llmCalls`, `admittedRules`).
- **Typed semantic policy (`.nl` leaves):**
  - `transformFile.nl`: transforms one uncovered source file given `MigrationSpec` and verifier `problems`, returning the updated source text.
  - `mineCrispRule.nl`: inspects verified before/after semantic migration pairs and proposes a reusable `{ find, replace, description, regex? }` crisp rule.

## Status and limits

- **Which files need migrating is decided by the verifier.** Stage 1 sends a file to the semantic stage only when the
  crisp draft fails `verify`. A file that still uses the old API but type-checks and passes its tests is reported
  `unchanged`, so `verify` must also reject files the migration has not reached (for example a crisp "old API still
  imported" check). `applications/migration` makes that decision in natural language (survey, site classification,
  per-site edit decisions, repair with an NL stopping judgment); plan 06 names it as this flagship's seed, and the
  semantic stage here (`transformFile.nl` inside `proposeAndVerify`) duplicates its edit/validate/repair loop instead
  of calling it. What this package adds is the crisp-rule pass and rule mining.
- **No fixture pack.** The earlier `fixtures/cases.json` named migrations and expected counts without any source
  files, so nothing could run it; it was removed.

## Safety & Custody

- Every repository file body crosses the boundary as `Untrusted<string>`.
- Unverified semantic proposals are never accepted into the workspace folder.
- Mined rules are never admitted without passing exact verification with zero regressions.
