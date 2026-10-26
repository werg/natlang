# Supplier incident coordinator — semantic decomposition

**Brief:** `applications/04-procurement-and-supplier-management/018-supplier-incident-coordinator.md`
**Domain:** `Procurement and supplier management`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `linkReports` → `IncidentCluster

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Compare complaints, inspections, and shipment events by product, lot, timing, and failure mechanism. Group plausibly related events while retaining independent incidents and contradictory evidence; do not merge all problems from one supplier into a single cause.
**Acceptance gate:** Cluster membership has a specific shared factor and preserves uncertain links.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `assessExposure` → `ExposureFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<IncidentCluster>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Trace potentially affected lots and orders through authoritative supply records. Distinguish confirmed exposure from precautionary scope, identify downstream commitments, and ask for missing traceability evidence before narrowing a containment boundary.
**Acceptance gate:** Exact lineage traversal; no unsupported exclusion of potentially affected stock.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `planContainment` → `ContainmentPlan

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExposureFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Select approved containment actions proportionate to the established issue and uncertainty. Coordinate holds, inspections, alternative supply, and reviewed communications while preserving unaffected orders and avoiding promises before capacity is reserved.
**Acceptance gate:** Actions remain within delegated authority and have explicit release criteria.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `interpretCorrection` → `CorrectiveAction

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ContainmentPlan>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Evaluate the supplier’s explanation against incident evidence and rival mechanisms. Separate immediate correction from prevention of recurrence, identify testable claims, and request evidence that the proposed change addresses the actual mechanism rather than just the symptoms.
**Acceptance gate:** A plausible narrative alone cannot close the causal investigation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyClosure` → `IncidentClosure

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CorrectiveAction>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Check implemented actions, release criteria, and subsequent observations. Preserve residual risk and monitoring obligations; reopen the incident when a materially similar recurrence contradicts the closure rationale rather than treating it as a fresh unrelated ticket.
**Acceptance gate:** All containment and verification obligations resolved or explicitly transferred to an accepted owner.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A supplier renumbers a lot: retain alias mapping and original traceability evidence.
- An order is split across affected and unaffected stock: release only the verified portion.
- The supplier misses a response deadline: escalate without automatically accepting the proposed explanation.
- A recurrence occurs after closure: link it to the prior causal claim and reopen the relevant verification.
- A customer notification is corrected: preserve delivery history and issue a reviewed amendment.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
