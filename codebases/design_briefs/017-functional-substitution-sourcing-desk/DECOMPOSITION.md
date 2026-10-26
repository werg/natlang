# Functional substitution sourcing desk — semantic decomposition

**Brief:** `applications/04-procurement-and-supplier-management/017-functional-substitution-sourcing-desk.md`
**Domain:** `Procurement and supplier management`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverFunction` → `FunctionalRequirement

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret why the original item is used, including interfaces, operating environment, and failure consequences. Separate critical properties from historical purchasing habits, and ask the requester to confirm distinctions that determine compatibility or safety.
**Acceptance gate:** Every essential property has a use-based rationale; unknown critical properties block qualification.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `generateCandidates` → `SubstituteCandidate

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<FunctionalRequirement>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Search catalogs and supplier descriptions for equivalent functions, including alternative mechanisms where permitted. Explain how each candidate would serve the use, identify required adaptations, and avoid treating a familiar name or matching dimension as sufficient.
**Acceptance gate:** Candidate generation cannot relax essential constraints or invent supplier specifications.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `assessCompatibility` → `CompatibilityFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SubstituteCandidate>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare every essential requirement against documented candidate evidence and approved conversion rules. Check interacting properties and interface tolerances; preserve unknowns explicitly and reject candidates whose apparent equivalence depends on an unsupported assumption.
**Acceptance gate:** All essential properties pass independently; one critical failure disqualifies the candidate.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `requestQualification` → `QualificationTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CompatibilityFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Design the smallest supplier clarification, sample inspection, or approved test that resolves a decisive unknown. Specify pass criteria before results arrive and distinguish documentary clarification from testing requiring technical specialists.
**Acceptance gate:** Test authority and acceptance criteria are explicit; no improvised hazardous test procedure.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `packageSubstitution` → `SubstitutionPacket

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<QualificationTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare qualified alternatives on verified cost, lead time, and operational consequences. Present the recommended substitution with adaptations and residual uncertainty, obtain technical and commercial approval, and record which uses the approval does not cover.
**Acceptance gate:** Approval is scoped to intended use, lot or revision where relevant, and current evidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A supplier silently changes a revision: invalidate compatibility findings tied to the old revision.
- Units differ across catalogs: use exact conversions and preserve original values.
- A sample passes but production lots vary: require the approved lot-level acceptance rule.
- A cheaper candidate needs an adapter: include adapter cost, lead time, and interface risk.
- A substitution is approved for one machine: prevent its approval from propagating to unrelated uses.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
