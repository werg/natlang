# Requirements negotiation workspace — semantic decomposition

**Brief:** `applications/02-customer-research-and-product-intelligence/010-requirements-negotiation-workspace.md`
**Domain:** `Customer research and product intelligence`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `extractPositions` → `StakeholderPosition

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read each request with its rationale and authority. Separate desired outcomes, proposed solutions, non-negotiable constraints, and preferences; do not upgrade emphatic language into a mandatory requirement without an explicit basis.
**Acceptance gate:** Each strength classification has evidence or remains unknown.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `modelConflicts` → `RequirementConflict

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<StakeholderPosition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Identify requirements that cannot jointly hold under the supplied technical, time, or resource constraints. Distinguish true contradictions from terminology differences and propose the smallest clarification or relaxation that would make the pair compatible.
**Acceptance gate:** Exact conflicts have a constraint witness; semantic conflicts retain explanatory counterexamples.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `designCompromises` → `CompromiseOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RequirementConflict>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Invent alternatives such as phased delivery, differentiated modes, or reduced scope while preserving mandatory constraints. Explicitly identify who gives up what and which new assumptions the compromise introduces; include a no-agreement option.
**Acceptance gate:** No concession is represented as accepted until the affected stakeholder confirms it.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `writeSpecification` → `Requirement

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CompromiseOption>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Write atomic, testable requirements from accepted positions and compromises. Define terms, negative requirements, and failure behavior; keep unresolved choices outside the committed baseline and link every clause to its negotiated source.
**Acceptance gate:** No orphan requirement, hidden assumption, or mutually inconsistent accepted clause.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `replayAcceptance` → `AcceptanceScenario

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<Requirement>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Generate ordinary, boundary, and adverse scenarios, then replay the specification through the available simulator. Use counterexamples to reopen only the implicated agreements, preserving unrelated acceptance and recording revised stakeholder acknowledgment.
**Acceptance gate:** Every mandatory requirement has a discriminating scenario and no blocking scenario remains unexplained.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A stakeholder changes role: retain historical assent and require successor authority for future changes.
- A requirement is verbally agreed but not recorded: keep it pending and issue a focused confirmation request.
- One compromise changes another team’s interface: require affected-party review before baseline acceptance.
- An acceptance test encodes an implementation preference: separate the outcome from the proposed method.
- Delivery is phased: tag each requirement with the release in which it becomes binding.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
