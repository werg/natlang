# Multi-party sourcing negotiator — semantic decomposition

**Brief:** `applications/04-procurement-and-supplier-management/020-multi-party-sourcing-negotiator.md`
**Domain:** `Procurement and supplier management`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `modelPackage` → `SourcingRequirement

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Translate the sourcing objective into functional, quantity, timing, and service requirements. Identify which terms may be traded and which are binding, and represent dependencies that make individual line-item offers valuable only as a package.
**Acceptance gate:** Hard constraints and delegated negotiation limits are separate and explicit.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `normalizeOffers` → `SupplierOffer

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SourcingRequirement>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile quotations into comparable units and scopes while preserving bundles, exclusions, and conditional discounts. Distinguish an indicative quote from a binding offer and expose missing service or logistics costs rather than assuming comparability.
**Acceptance gate:** Exact totals and quote validity; no hidden removal of supplier conditions.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `constructPackages` → `SourcingPackage

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SupplierOffer>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Use exact allocation services to enumerate feasible combinations, then compare operational resilience and semantic fit under supplied priorities. Consider split supply and alternative delivery patterns without creating allocations that suppliers have not offered capacity to fulfill.
**Acceptance gate:** Quantity conservation, capacity limits, and conditional-offer compatibility all pass.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `formCounteroffers` → `Counteroffer

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SourcingPackage>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft conditional counteroffers that exchange permitted concessions for improvements in the whole package. Keep supplier-confidential information isolated and make dependencies explicit so acceptance of one term cannot accidentally commit the buyer to an unapproved package.
**Acceptance gate:** No disclosure of rival bids or offer beyond the signed negotiation mandate.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `reconcileAgreement` → `SourcingAgreement

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<Counteroffer>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret responses as acceptance, rejection, or qualified counteroffer at the term level. Rebuild package feasibility after each material change, identify unresolved conditions, and prepare a final agreement only when all linked commitments are mutually consistent.
**Acceptance gate:** Qualified acceptance is not final acceptance; exact final package must match approvals.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- One supplier’s discount requires exclusivity: invalidate split-supply packages using that discount.
- An offer expires overnight: revalidate before presenting it as available.
- A salesperson accepts only verbally: retain provisional status pending the configured evidence.
- A currency rate changes: freeze the approved conversion basis and show sensitivity separately.
- A buyer changes the mandate: stop outstanding unapproved offers and review their external status.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
