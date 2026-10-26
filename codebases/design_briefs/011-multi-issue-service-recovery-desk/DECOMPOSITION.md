# Multi-issue service recovery desk — semantic decomposition

**Brief:** `applications/03-customer-service-and-commercial-casework/011-multi-issue-service-recovery-desk.md`
**Domain:** `Customer service and commercial casework`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `decomposeCase` → `ServiceIssue

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Reconstruct the timeline across messages and split failures only when they can be resolved independently. Retain earlier promises and interactions between issues, such as a late replacement making a promotional remedy no longer useful.
**Acceptance gate:** Every issue links to the relevant order and the customer’s stated outcome.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `interpretEntitlements` → `EntitlementFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ServiceIssue>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Match each issue to the applicable policy edition, purchased service, and documented exception. Separate mandatory entitlements from discretionary gestures; ask for missing decisive evidence without requiring information already present in the case.
**Acceptance gate:** No remedy outside delegated authority; original purchase dates determine applicable terms.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `composeRecovery` → `RemedyBundle

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<EntitlementFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct alternative bundles that jointly solve the issues, accounting for dependencies and customer preferences. Detect double compensation for the same loss, but do not erase separate entitlements merely because they share one order.
**Acceptance gate:** Exact amount limits and mutually exclusive remedies checked before presentation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `coordinateActions` → `RecoveryAction

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RemedyBundle>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Sequence approved refunds, replacements, pickups, and communications. Reserve scarce replacements before promising dates, wait for required customer choices, and reconcile uncertain external outcomes rather than retrying irreversible operations blindly.
**Acceptance gate:** Every effect has an operation identity and a matching approval or preauthorized policy grant.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyClosure` → `RecoveryClosure

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RecoveryAction>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare actual outcomes with every agreed remedy and earlier promise. A refund request or shipment label is not completion; retain unresolved components and explain partial closure without resetting the entire case history.
**Acceptance gate:** Close only when all mandatory outcomes are evidenced or an explicit alternative is accepted.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A refund succeeds but the network response is lost: reconcile by operation ID before any retry.
- A replacement goes out of stock during consent: expire the reservation and renegotiate the promise.
- A supervisor grants an exception: bind it to this customer, issue, amount, and expiry.
- The customer changes channels: preserve one case state while respecting contact preferences.
- One issue resolves early: communicate partial completion without hiding the remaining obligation.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
