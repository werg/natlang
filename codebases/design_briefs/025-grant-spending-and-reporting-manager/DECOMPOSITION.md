# Grant spending and reporting manager — semantic decomposition

**Brief:** `applications/05-finance-and-administrative-accounting/025-grant-spending-and-reporting-manager.md`
**Domain:** `Finance and administrative accounting`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `extractGrantRules` → `GrantCondition

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Parse the agreement, budget schedules, and amendments into spending, milestone, and reporting conditions. Distinguish required outcomes from aspirational impact language, and preserve restrictions that apply only to specific funding tranches or time periods.
**Acceptance gate:** Every condition has effective scope and an attributable grant source.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `allocateSpending` → `GrantAllocation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<GrantCondition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Match expenditures to eligible activities using descriptions, service periods, and supporting records. Split shared costs only under documented allocation rules, detect duplicate funding attribution, and retain unsupported portions for review.
**Acceptance gate:** Exact amount conservation and no double attribution across grants.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `verifyActivities` → `ActivityEvidence

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<GrantAllocation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare project records with milestone definitions and acceptance evidence. Distinguish activity completion from demonstrated impact, preserve partial delivery, and request missing evidence rather than embellishing a narrative to fit the grant’s aspirations.
**Acceptance gate:** Claimed outputs and outcomes are separately supported.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `composeReport` → `GrantNarrative

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ActivityEvidence>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft a coherent account aligned with actual spending, activities, and verified results. Explain deviations and unresolved conditions honestly, tailor the format to funder requirements, and cross-check narrative quantities against exact financial schedules.
**Acceptance gate:** No unsupported impact claim or mismatch between narrative and financial totals.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `manageSubmission` → `GrantSubmission

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<GrantNarrative>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Assemble the required forms, schedules, and evidence, obtain accountable approval, and record actual submission receipt. Track follow-up questions as obligations and revise reports through explicit versions rather than silently replacing the submitted record.
**Acceptance gate:** Required deliverables and approvals complete; submission receipt and revision history retained.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A cost spans two reporting periods: allocate using the approved basis and preserve service dates.
- A grant amendment arrives late: distinguish retroactive permission from future-only changes.
- The same activity supports two grants: prevent duplicate attribution while allowing documented shared funding.
- A funder requests personal participant data: route disclosure through the consent and authority check.
- A report is corrected after submission: issue a traceable amended version.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
