# Accounts-payable exception resolver — semantic decomposition

**Brief:** `applications/05-finance-and-administrative-accounting/021-accounts-payable-exception-resolver.md`
**Domain:** `Finance and administrative accounting`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `matchDocuments` → `InvoiceMatch

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Align descriptions, units, supplier references, and delivery context across invoice and order lines. Permit many-to-many matches for split shipments, but preserve ambiguous allocations and do not force every invoice line onto the nearest description.
**Acceptance gate:** Exact unit conversions and sum conservation; candidate links retain provenance.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `classifyDiscrepancy` → `PayableDiscrepancy

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<InvoiceMatch>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret differences using contract terms and receiving evidence. Distinguish partial delivery, unauthorized substitution, freight charges, and genuine duplication; a repeated invoice number is a signal to investigate rather than a sufficient payment decision.
**Acceptance gate:** Every disputed amount derives from exact line calculations and a named discrepancy.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `resolveAllocation` → `AllocationProposal

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PayableDiscrepancy>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct a partial resolution for portions supported by delivery and authorization evidence. Preserve disputed balances, prevent the same receipt from supporting multiple payments, and request targeted confirmation for unresolved acceptance or scope.
**Acceptance gate:** Accepted plus held amounts equal the invoice balance after verified credits.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `prepareDisposition` → `PayableDisposition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AllocationProposal>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Prepare an auditable recommendation to pay, hold, request correction, or apply a verified credit. Route by entity and delegated thresholds, and explain why undisputed portions may proceed without treating the whole invoice as resolved.
**Acceptance gate:** Segregation of duties and current approvals enforced outside the semantic executor.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `reconcilePosting` → `PostingReceipt

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PayableDisposition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare approved allocations with actual ledger and processor results. Reconcile lost acknowledgments by operation identity, detect subsequent credit notes, and maintain remaining obligations without retrying a payment whose outcome is unknown.
**Acceptance gate:** No duplicate posting; ledger balance and approved allocation agree exactly.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A supplier sends a corrected invoice with the same number: version documents without assuming a second liability.
- A receipt is reused across invoices: reserve its allocatable quantity transactionally.
- A credit arrives after approval: invalidate the affected payment amount before release.
- Currencies differ: require an explicit authorized conversion basis.
- The period closes during review: preserve original service dates and route posting-period changes for approval.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
