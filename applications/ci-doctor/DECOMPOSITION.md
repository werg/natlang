# CI Doctor (`applications/ci-doctor/`): Decomposition

Tier 2 flagship A5 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3): triages CI failure traces, clusters
root causes, bisects regression commits, and selects verified patches without allowing test-assertion weakening.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `doctor.ts`):**
  - `detectKnownFlakeSignature` wrapped in `pluggable` (`default: 'crisp'`): settles well-known network/runner flake
    signatures (`ECONNRESET`, `ETIMEDOUT`, `socket hang up`, `HTTP 429`, `503`) with zero model calls; its nl side is
    `classifyFailure.nl` answering `flake`, and `shadow` runs both and serves crisp. A log that also shows a deterministic failure
    (`AssertionError`, `TypeError`, `error TS…`) is never settled crisply.
  - `fenceTrace`: wraps every external CI log in `untrusted(...)` before crossing any `.nl` boundary.
  - `semanticPartition`: classifies remaining traces concurrently by `FailureKind` (`'flake' | 'env' | 'regression'`)
    and isolates low-confidence items into `uncertain`.
  - `clusterFailures`: within each kind, std `semanticMatch` asks of each pair whether the two failures have the same
    root cause (shipped `relates.nl`), and the pairs join transitively into clusters (`clustering: 'signature'` groups
    by the exact first error line instead). Known flake signatures stay exact clusters.
  - `bisectCommits`: the step is an exact binary search over `commits`, bounded by `iterateOn(...).withMeasure(...)`;
    the stop is a `pluggable` (`bisectStop`, default `nl`): the range narrowed to one commit, or `bisectSettled.nl`
    judging from the tested commits' outputs (`failsAtCommit` may return `{ fails, log }`) that going on cannot find
    this failure's cause (another breakage, an untestable commit). A judged stop reports `bisectStopped: 'judged'` and
    blames no commit.
  - `checkAssertionsPreserved`: exact verifier counting `assert(...)` / `expect(...)` calls and `.skip(...)` markers
    across every changed file in a `FolderProposal`, rejecting any patch that deletes, skips, or edits an existing
    assertion line (an edited assertion counts as weakened).
  - `branchAndSelect`: runs candidate fixes on private `Folder` copies, rejects any candidate failing
    `checkAssertionsPreserved` or `runTests`, and accepts the winning verified fix.
- **Typed semantic policy (`.nl` functions):**
  - `bisectSettled.nl` (`readout: decision`, returns `boolean`): whether the bisection should stop before it has
    narrowed the range, read from the tested commits' outcomes and outputs.
  - `classifyFailure.nl` (`readout: decision`, returns `FailureKind`): classifies untrusted CI logs into
    `'flake' | 'env' | 'regression'`.
  - `proposeFix.nl` (`kind: directory-reducer`, returns `FixProposal`): proposes a targeted source-code fix in the
    repository `Folder` while keeping all test assertions intact.
