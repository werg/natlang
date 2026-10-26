# Financial close workpaper coordinator — semantic decomposition

**Brief:** `applications/05-finance-and-administrative-accounting/024-financial-close-workpaper-coordinator.md`
**Domain:** `Finance and administrative accounting`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `frameReconciliation` → `ReconciliationScope

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret the purpose and supplied close policy for each workpaper. Identify the assertion to establish, relevant systems, exclusions, and escalation thresholds; keep documentary preparation separate from decisions reserved for qualified accounting reviewers.
**Acceptance gate:** Scope and source snapshots are explicit before analysis begins.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `classifyBreaks` → `ReconciliationBreak

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReconciliationScope>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Consume exact reconciliation differences and inspect supporting narratives. Distinguish timing, mapping, missing postings, and unsupported explanations; group related breaks only when they share a demonstrated mechanism rather than the same amount.
**Acceptance gate:** Numerical equality alone does not establish a valid match or explanation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `commissionEvidence` → `CloseTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReconciliationBreak>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Translate each consequential unresolved break into a bounded investigation with an accountable owner. Specify evidence capable of proving the explanation, prioritize dependencies on other close tasks, and expose unresolved issues before sign-off deadlines.
**Acceptance gate:** Every material break has an accepted owner and completion criterion.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `composeWorkpaper` → `WorkpaperConclusion

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CloseTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Assemble the reconciliation, explanations, and supporting documents into a reviewer-ready workpaper. Distinguish verified facts from management explanations, quantify residual differences exactly, and avoid conclusions broader than the evidence supports.
**Acceptance gate:** Totals reconcile; residual material issues remain visible and block unqualified completion.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `maintainSignoff` → `WorkpaperSignoff

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<WorkpaperConclusion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Bind sign-off to exact source and conclusion versions. When a late journal or source correction arrives, trace affected assertions and invalidate only dependent approvals; preserve the prior signed workpaper as historical evidence.
**Acceptance gate:** No current sign-off over stale inputs; reviewer independence enforced by the host.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A late journal changes one account: reopen only dependent workpapers and show the exact delta.
- A source report is regenerated with identical totals but changed lines: compare content hashes, not totals alone.
- A preparer leaves mid-close: preserve authorship and reassign outstanding tasks explicitly.
- A spreadsheet formula is broken: quarantine its computed outputs until exact validation passes.
- A deadline is missed: record the unresolved state rather than backdating sign-off.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
