# Replication planning workbench — semantic decomposition

**Brief:** `applications/07-scientific-evidence-and-research-operations/033-replication-planning-workbench.md`
**Domain:** `Scientific evidence and research operations`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverProcedure` → `MethodStep

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Reconstruct the reported procedure from methods, supplements, code descriptions, and notes. Separate explicitly specified parameters from inferred defaults and distinguish the authors’ actual method from a convenient modern replacement.
**Acceptance gate:** Every decisive methodological choice has a source or remains an explicit unknown.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `mapDependencies` → `ReplicationDependency

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MethodStep>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Identify data, software, configuration, and computational dependencies for each step. Detect unavailable versions and incompatible assumptions, then formulate approved alternatives without representing them as exact replication of the original environment.
**Acceptance gate:** Dependency graph is acyclic and all required resources are available or blocked.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `resolveAmbiguities` → `MethodQuestion

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReplicationDependency>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Enumerate plausible interpretations of missing details and assess whether each changes the result. Seek author clarification or documented evidence for consequential choices; retain sensitivity branches where ambiguity cannot be resolved.
**Acceptance gate:** No unacknowledged methodological invention or selective choice after seeing favorable results.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `compileExecution` → `ReplicationTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MethodQuestion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Translate the reviewed procedure into typed tasks using approved computational capabilities. Pin inputs and environments, define checkpoint and output contracts, and include resource limits and recovery behavior for failed jobs.
**Acceptance gate:** Commands execute only in the sandbox and consume the pinned dependency manifest.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `compareOutcomes` → `ReplicationFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReplicationTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare outputs with the original reported quantities and tolerances, distinguishing implementation failures from substantive differences. Investigate divergences through minimal controlled changes and publish uncertainty rather than claiming replication from superficial similarity.
**Acceptance gate:** Exact comparisons and methodological provenance support every reported match or discrepancy.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A dependency version is unavailable: label the substitute as an adaptation and preserve the original requirement.
- A random seed is omitted: run a preregistered sensitivity set rather than choosing a favorable seed.
- A dataset has changed: record the mismatch and do not claim exact replication.
- A job crashes after producing partial outputs: verify completeness before reuse.
- A paper’s prose and code disagree: retain both interpretations for review.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
