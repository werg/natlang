# Exception and precedent steward — semantic decomposition

**Brief:** `applications/01-organizational-knowledge-and-institutional-memory/005-exception-and-precedent-steward.md`
**Domain:** `Organizational knowledge and institutional memory`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `normalizeExceptions` → `ExceptionRecord

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Extract the exact deviation, beneficiary, dates, and conditions of each granted exception. Keep the requirement that was waived separate from prerequisites that remained binding, and record whether the decision explicitly denied precedential value.
**Acceptance gate:** An exception never inherits broader scope than its authorization.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `retrievePrecedents` → `PrecedentCandidate

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExceptionRecord>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Search by the rationale and relevant circumstances rather than matching requested outcomes. Construct analogies and disanalogies, including expired and rejected cases when they explain the boundary of acceptable treatment.
**Acceptance gate:** Retrieval includes a negative precedent slice; similarity alone cannot authorize a waiver.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `judgeApplicability` → `ApplicabilityFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PrecedentCandidate>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Test every limiting condition against current evidence and identify which differences change the rationale. If a decisive fact is unknown, request that fact; do not grant the exception by treating missing evidence as a match.
**Acceptance gate:** All mandatory conditions established and current authority confirmed before recommendation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `formDecision` → `ExceptionDisposition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ApplicabilityFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct a scoped recommendation with the original rule, proposed deviation, continuing controls, and expiration. Explain consistency with comparable cases and explicitly surface any departure requiring a higher authority.
**Acceptance gate:** Required reviewer and expiration present; surviving safeguards not accidentally waived.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `reviewPatterns` → `PolicyReviewSignal

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExceptionDisposition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Group substantively similar exceptions while discounting copied requests and a single repeated incident. Formulate a policy revision and boundary examples, then simulate historical outcomes before referring the proposal to the policy owner.
**Acceptance gate:** Frequency is not automatic justification; proposed revision passes supplied counterexample cases.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An exception has no expiry: flag a governance defect rather than interpreting it as perpetual.
- A private precedent applies: disclose the rationale only to authorized reviewers.
- The policy changes while approval waits: re-evaluate applicability before signing.
- A rejected request is resubmitted with new wording: link identity without assuming unchanged facts.
- A waiver is revoked: notify dependent owners and preserve prior valid actions in history.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
