# Delivery-to-billing obligation reconciler — semantic decomposition

**Brief:** `applications/05-finance-and-administrative-accounting/022-delivery-to-billing-obligation-reconciler.md`
**Domain:** `Finance and administrative accounting`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `extractConditions` → `BillingCondition

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read contract schedules and amendments to identify triggers, acceptance requirements, partial billing rules, and exclusions. Distinguish work performed from the specific event authorizing billing and retain conditions that apply only to particular deliverables.
**Acceptance gate:** Every billable condition has a source and effective scope.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `mapDelivery` → `DeliveryEvidence

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<BillingCondition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Relate project artifacts, service logs, and customer messages to the promised deliverables. Identify partial or alternative delivery and distinguish a draft shared for comment from an accepted contractual handover.
**Acceptance gate:** Evidence supports the actual milestone, not merely related activity.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `assessAcceptance` → `AcceptanceFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DeliveryEvidence>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret customer responses under the agreement’s acceptance process. Separate acknowledgment of receipt from acceptance, verify the respondent’s authority, and apply deemed-acceptance timing only when all supplied contractual prerequisites are established.
**Acceptance gate:** No implied acceptance outside the configured agreement; exact date calculations use host services.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `calculateBillability` → `BillableAllocation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AcceptanceFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Submit established milestone states and exact schedules to the billing calculator. Account for prior invoices, credits, caps, and partial acceptance; explain remaining holds without inventing a commercial compromise not authorized by the contract.
**Acceptance gate:** No overbilling, duplicate milestone consumption, or amount based on an unresolved trigger.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `assemblePacket` → `BillingPacket

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<BillableAllocation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Assemble a customer-readable explanation and internal approval packet linking each amount to satisfied conditions. Recheck late amendments and acceptance changes before release, then reconcile the actual invoice identifier back to consumed obligations.
**Acceptance gate:** Approved packet hash, current conditions, and exact ledger reconciliation required.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A customer accepts only one phase: allocate partial billing without marking the entire project complete.
- A milestone is renamed in project software: preserve the contract identity.
- Acceptance comes from an unauthorized contact: seek confirmation rather than discarding the message.
- An amendment changes future rates: retain prior earned amounts under their effective schedule.
- An invoice is voided: release consumed billing capacity only after authoritative confirmation.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
