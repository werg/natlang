# Handover continuity manager — semantic decomposition

**Brief:** `applications/01-organizational-knowledge-and-institutional-memory/003-handover-continuity-manager.md`
**Domain:** `Organizational knowledge and institutional memory`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverCommitments` → `Commitment

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read threads with their replies and amendments. Distinguish a tentative intention from a promise, identify the beneficiary and deliverable, and retain conditional commitments whose activation event has not yet occurred.
**Acceptance gate:** Every commitment has an attributable statement; missing dates stay unknown rather than invented.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `reconstructStatus` → `WorkStatus

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<Commitment>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Match outputs and acknowledgments to each promised outcome. A sent file is not accepted delivery when acceptance was required; cancellation must come from an authorized party, and incomplete evidence produces a status question.
**Acceptance gate:** Completion requires the specified evidence, not merely a past due date or optimistic message.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `mapDependencies` → `HandoverDependency

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<WorkStatus>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Identify knowledge, access, approvals, and external relationships needed to finish each item. Separate credentials from instructions, detect circular handoffs, and formulate minimal access requests without copying secrets into the handover dossier.
**Acceptance gate:** No plaintext credentials; every blocked dependency has an owner and a recovery task.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `allocateOwnership` → `OwnershipTransfer

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<HandoverDependency>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare successor remit, stated capacity, and required capabilities. Propose explicit ownership with acceptance criteria and unresolved obligations; a recipient mentioned in a meeting is not treated as accepting the transfer.
**Acceptance gate:** Transfer remains pending until the named successor acknowledges the current scope.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyContinuity` → `ContinuityReview

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<OwnershipTransfer>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Revisit the portfolio after departure, follow missing acknowledgments, and compare completed artifacts to promises. Close the handover only when ownership and immediate next steps are established, while operational work may legitimately remain open.
**Acceptance gate:** Handover completion and task completion are separate statuses with separate gates.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An employee loses access before departure: freeze authorized evidence snapshots without retaining unnecessary personal mail.
- A successor accepts only part of a project: split obligations before acknowledging ownership.
- An external customer expects a named person: draft a reviewed introduction and record delivery.
- A recurring task has no written deadline: derive a proposed cadence but require owner confirmation.
- A personal note contains useful context: request selective sharing rather than exporting the whole notebook.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
