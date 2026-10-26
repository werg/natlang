# Contract-to-purchasing conformance desk — semantic decomposition

**Brief:** `applications/04-procurement-and-supplier-management/019-contract-to-purchasing-conformance-desk.md`
**Domain:** `Procurement and supplier management`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `interpretRequest` → `PurchaseIntent

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read item descriptions with the requester’s purpose and supporting documents. Distinguish a legitimate functional need from a preferred brand or category label, and identify ambiguity that changes contract coverage or approval authority.
**Acceptance gate:** Quantity and purpose are explicit; missing critical scope is not guessed.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `selectAgreement` → `AgreementScope

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PurchaseIntent>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Resolve the applicable contract, schedules, amendments, and purchasing entity. Interpret coverage under dates, territory, and intended use; do not apply a parent agreement to an affiliate or service category without supporting terms.
**Acceptance gate:** Exact temporal and entity scope plus supported semantic category match.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `judgeConformance` → `ConformanceFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AgreementScope>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare each requested line with contract inclusions, exclusions, and conditions. Separate commercial price compliance from functional scope and preserve conditions such as minimum quantities, required bundles, or approved use limitations.
**Acceptance gate:** A catalog listing alone does not establish contractual coverage.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `proposeAlternatives` → `PurchasingOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ConformanceFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct alternatives that satisfy the underlying need within the applicable agreement. Explain changed capabilities and total costs, and offer an exception route where compliant options would not meet essential requirements rather than silently weakening the request.
**Acceptance gate:** Alternatives preserve confirmed essential constraints and exact commercial limits.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `routeApproval` → `PurchaseDisposition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PurchasingOption>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Build a review packet with scope evidence, pricing, alternatives, and unresolved conditions. Route to the correct authority based on cumulative commitment and exception type; recheck terms and request versions immediately before order release.
**Acceptance gate:** No split-order circumvention of thresholds; current scoped approval required.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A requester splits one purchase into several small requests: aggregate related commitments for threshold checks.
- A discount depends on a bundle: recalculate when one item is removed.
- An affiliate uses the parent’s contract: require explicit entity coverage.
- A catalog description changes: retain the version used for approval.
- A contract expires while approval waits: revalidate before releasing the order.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
