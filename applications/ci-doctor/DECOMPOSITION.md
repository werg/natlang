# CI Doctor (`applications/ci-doctor/`): Decomposition

Tier 2 flagship A5 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3): triages CI failure traces, clusters
root causes, bisects regression commits, and selects verified patches without allowing test-assertion weakening.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `doctor.ts`):**
  - `detectKnownFlakeSignature` wrapped in `pluggable` (`default: 'crisp'`): settles well-known network/runner flake
    signatures (`ECONNRESET`, `ETIMEDOUT`, `socket hang up`, `HTTP 429`, `503`) with zero model calls.
  - `fenceTrace`: wraps every external CI log in `untrusted(...)` before crossing any `.nl` boundary.
  - `semanticPartition`: classifies remaining traces concurrently by `FailureKind` (`'flake' | 'env' | 'regression'`)
    and isolates low-confidence items into `uncertain`.
  - `bisectCommits`: bounded binary search over `commits` using `iterateOn(...).withMeasure(...).withLimit(...)`.
  - `checkAssertionsPreserved`: exact verifier counting `assert(...)` / `expect(...)` calls and `.skip(...)` markers
    across every diff hunk in a `FolderProposal`, rejecting any patch that deletes, skips, or weakens assertions.
  - `branchAndSelect`: runs candidate fixes on private `Folder` copies, rejects any candidate failing
    `checkAssertionsPreserved` or `runTests`, and accepts the winning verified fix.
- **Typed semantic policy (`.nl` functions):**
  - `classifyFailure.nl` (`readout: decision`, returns `FailureKind`): classifies untrusted CI logs into
    `'flake' | 'env' | 'regression'`.
  - `proposeFix.nl` (`kind: directory-reducer`, returns `FixProposal`): proposes a targeted source-code fix in the
    repository `Folder` while keeping all test assertions intact.
