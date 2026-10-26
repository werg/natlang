# Returns, repair, and resale orchestrator — semantic decomposition

**Brief:** `applications/03-customer-service-and-commercial-casework/012-returns-repair-and-resale-orchestrator.md`
**Domain:** `Customer service and commercial casework`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `assessCondition` → `ConditionAssessment

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile customer descriptions, inspection notes, and supplied images or structured image annotations. Distinguish cosmetic damage, functional symptoms, and unverified allegations; preserve uncertainty and request evidence that changes routing rather than generic photo uploads.
**Acceptance gate:** No unsupported diagnosis or condition grade; evidence provenance remains visible.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `selectPath` → `ReturnDisposition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ConditionAssessment>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Apply purchase terms and approved product-handling rules to condition evidence and customer preference. Identify cases needing specialist inspection, compare repair and replacement options, and avoid promising a remedy before required checks are satisfied.
**Acceptance gate:** Route respects policy, customer consent, and approved handling restrictions.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `planRepair` → `RepairPlan

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReturnDisposition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Translate the selected path into authorized inspection or repair tasks using supplied procedures. Check part compatibility and workshop capacity, preserve stop conditions, and create alternative plans when repair would exceed delegated cost or capability.
**Acceptance gate:** Exact parts and capacity feasibility; no improvised hazardous repair instruction.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `reconcileCustody` → `CustodyEvent

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RepairPlan>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Track collection, receipt, repair transfer, and return as distinct custody events. Compare condition at each handoff and isolate new discrepancies rather than attributing all damage to the customer or overwriting the original inspection.
**Acceptance gate:** Inventory changes require a confirmed custody event; serial-number mismatches block transfer.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `prepareResale` → `ResaleDisclosure

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CustodyEvent>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Produce a truthful resale or disposal packet from verified work and remaining defects. Reconcile grade terminology with marketplace rules, remove prior-owner data, and hold listing until required functional checks and ownership transfer are complete.
**Acceptance gate:** Disclosure covers known defects; data-removal and title checks pass before listing.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- Two identical items have different serial numbers: block accidental swapping during repair intake.
- Personal data remains on a device: quarantine resale until approved erasure is verified.
- A customer disputes transit damage: preserve each custody observation and the dispute separately.
- A part substitution is proposed: require compatibility evidence before updating the repair plan.
- A return label expires: regenerate the logistics artifact without creating a duplicate return entitlement.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
