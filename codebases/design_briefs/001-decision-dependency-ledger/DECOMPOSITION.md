# Decision-dependency ledger — semantic decomposition

**Brief:** `applications/01-organizational-knowledge-and-institutional-memory/001-decision-dependency-ledger.md`
**Domain:** `Organizational knowledge and institutional memory`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `extractDecisions` → `DecisionAssertion

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Segment meetings at topic boundaries; distinguish a suggestion, assent, and an authorized resolution. Link explicit objections and preserve the speaker rather than attributing a collective decision from an unchallenged statement.
**Acceptance gate:** No accepted resolution without attributable acceptance and scope evidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `bindAssumptions` → `AssumptionBinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DecisionAssertion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** For each decision ask what would have to change to make reconsideration necessary. Separate essential premises from background rationale; emit a directed dependency with a counterfactual explanation and a passage establishing the connection.
**Acceptance gate:** Every edge has source support; inferred edges remain proposals until owner review.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `assessChange` → `ChangeImpact

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<AssumptionBinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare old and new assertions under their dates, scope, and authority. Determine which premise is actually invalidated and retain the part of the decision whose conditions still hold; never treat a synonym edit as a substantive reversal.
**Acceptance gate:** Exact interval overlap plus semantic premise contradiction, not keyword overlap alone.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `planReconsideration` → `ReviewPlan

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ChangeImpact>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Traverse only necessary dependencies first, then inspect supporting edges for materiality. Group reviews sharing an unresolved premise, formulate a decision-ready question, and avoid automatically cancelling downstream work that remains independently justified.
**Acceptance gate:** No dependency cycle silently flattened; each affected owner receives one deduplicated task.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `recordResolution` → `ResolutionDelta

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ReviewPlan>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile owner responses against the pending question and current evidence version. Distinguish approval of wording from approval of the decision; append the adopted resolution and invalidate only consumers of the replaced version.
**Acceptance gate:** Signed owner authority and expected version must match before commit.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A decision owner leaves: retain the historical author and route new reviews through an explicit successor assignment.
- A private source supports a public decision: expose a restricted-evidence marker, never a revealing excerpt.
- Two reviews finish concurrently: reject the stale resolution and show the intervening change rather than overwriting.
- A renamed project receives copied notes: keep identity mapping separate from evidence independence.
- A meeting correction arrives late: retain both recorded and effective dates and reopen only the affected interval.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
