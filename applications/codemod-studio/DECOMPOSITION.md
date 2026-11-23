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

## Safety & Custody

- Every repository file body crosses the boundary as `Untrusted<string>`.
- Unverified semantic proposals are never accepted into the workspace folder.
- Mined rules are never admitted without passing exact verification with zero regressions.
- Fixture rows in `fixtures/cases.json` use `split_group: "flagship:codemod-studio:<group>"`.
