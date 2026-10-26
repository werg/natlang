# Expense policy casework system — semantic decomposition

**Brief:** `applications/05-finance-and-administrative-accounting/023-expense-policy-casework-system.md`
**Domain:** `Finance and administrative accounting`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `normalizeClaim` → `ExpenseLine

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Extract receipt lines and claimant explanations while preserving original currency and dates. Distinguish the merchant’s description from the claimed business purpose, and flag illegible or missing evidence rather than reconstructing amounts through guesswork.
**Acceptance gate:** Exact totals reconcile to source receipts; unverified values remain blocked.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `interpretPurpose` → `PurposeAllocation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExpenseLine>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Identify mixed business and personal components using explicit facts and supplied allocation rules. Request a targeted explanation when purpose changes eligibility, and avoid inferring personal behavior from merchant stereotypes or unrelated account history.
**Acceptance gate:** Shares sum exactly to one; allocation has evidence or remains pending.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `applyPolicy` → `ExpenseFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PurposeAllocation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Apply the policy edition effective for the expense and claimant context. Preserve category-specific limits, meal inclusions, and approval conditions; distinguish a documented exception from a persuasive but unauthorized justification.
**Acceptance gate:** Exact category and period limits; no retroactive policy substitution without authority.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `requestResolution` → `ExpenseQuestion

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExpenseFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Group unresolved questions to minimize claimant burden while keeping each decision testable. Ask only for evidence necessary to determine eligibility, explain the applicable rule, and route privacy-sensitive material through restricted upload channels.
**Acceptance gate:** Do not request information already available or unrelated personal records.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `prepareExpenseDisposition` → `ExpenseDisposition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExpenseQuestion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Produce line-level recommendations and a clear explanation of reductions or holds. Support independent review and appeals with new evidence, then reconcile the approved amount to the reimbursement service without changing the historical original claim.
**Acceptance gate:** Segregated approval and exact total conservation across paid, rejected, and held portions.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A receipt includes a personal companion: apply explicit allocation rather than rejecting the whole trip.
- A tax tip or fee is separate: preserve line-level amounts and the relevant rule.
- A claimant resubmits a rejected receipt: link the appeal without creating duplicate reimbursement.
- The policy changes during a trip: evaluate the supplied effective-date rule.
- A manager is also the claimant: route approval to an independent authority.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
