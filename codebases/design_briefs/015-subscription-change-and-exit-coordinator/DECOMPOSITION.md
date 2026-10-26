# Subscription change and exit coordinator — semantic decomposition

**Brief:** `applications/03-customer-service-and-commercial-casework/015-subscription-change-and-exit-coordinator.md`
**Domain:** `Customer service and commercial casework`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `interpretIntent` → `ChangeIntent

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read the request in conversational context and identify the intended account scope. Distinguish cancelling an add-on from closing the account, and ask a focused question only when ambiguity would change access, billing, or data retention.
**Acceptance gate:** No retention offer or unrelated account action is inferred as consent.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `computeConsequences` → `ChangeConsequence

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ChangeIntent>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Consult authoritative entitlement and billing services for permitted dates and amounts, then explain consequences in plain language. Separate immediate access effects from future billing and data deletion; keep uncertain external balances explicitly unresolved.
**Acceptance gate:** Monetary values come from exact services; all material consequences are disclosed.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `constructChoices` → `ChangeOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ChangeConsequence>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Form a small set of permitted choices centered on the requested outcome. Include the direct exit path without manipulative friction, disclose linked-service effects, and avoid presenting an unrequested alternative as the default selection.
**Acceptance gate:** Every option is feasible and the originally requested permitted action remains accessible.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `executeSelection` → `SubscriptionOperation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ChangeOption>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Verify consent to the exact option and effective date, then apply changes through idempotent adapters. Recheck account versions before commitment and reconcile partial changes across linked products using compensating actions where permitted.
**Acceptance gate:** No changed selection after consent; unknown outcomes block blind repetition.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyExit` → `ExitReceipt

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SubscriptionOperation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare actual subscription states and upcoming invoices with the selected option. Explain surviving obligations and access explicitly, provide a durable receipt, and reopen inconsistencies rather than declaring success after merely submitting a cancellation request.
**Acceptance gate:** All affected subscriptions match the confirmed choice; no unapproved service remains active.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A cancellation arrives near renewal: use authoritative timestamps and the applicable cutoff rule.
- One product is managed by a reseller: expose the separate required action rather than claiming full cancellation.
- A customer changes their mind before execution: invalidate the old consent token.
- A refund is pending: do not hide it behind a closed subscription status.
- A shared workspace has other owners: require the appropriate authority before changing their access.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
