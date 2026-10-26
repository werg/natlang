# Tender response factory — semantic decomposition

**Brief:** `applications/04-procurement-and-supplier-management/016-tender-response-factory.md`
**Domain:** `Procurement and supplier management`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `parseObligations` → `TenderRequirement

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read the main tender, schedules, forms, and addenda together. Extract explicit requirements and conditional submission instructions, distinguish mandatory eligibility from scored preferences, and preserve exact formatting or attachment obligations separately from substantive answers.
**Acceptance gate:** Every source requirement maps to a response slot or an explicit nonresponse decision.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `mapCapabilities` → `CapabilityMatch

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<TenderRequirement>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare required outcomes with verified capabilities and their scope. Do not stretch an adjacent feature into compliance; identify exceptions, delivery dependencies, and evidence that would substantiate a currently unverified claim.
**Acceptance gate:** Unsupported certifications and future capabilities cannot be represented as current compliance.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `commissionInputs` → `ContributionTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CapabilityMatch>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Group missing technical, financial, and implementation contributions by responsible team. Ask precise questions tied to requirements, specify acceptable evidence, and expose dependencies that could make the submission infeasible before the deadline.
**Acceptance gate:** Every mandatory gap has an accountable owner or an explicit bid/no-bid escalation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `composeResponse` → `ResponseSection

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ContributionTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft consistent answers using supported claims and approved commercial positions. Tailor explanations to the customer’s evaluation criteria without inventing facts; preserve declared exceptions and reconcile promises across technical, pricing, and delivery sections.
**Acceptance gate:** Each material claim has evidence or a clearly labeled future commitment approved by its owner.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `validateSubmission` → `SubmissionManifest

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ResponseSection>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Re-run coverage and cross-section consistency, validate file naming and required forms, and preview the exact bundle. Freeze approvals to its hash, submit through an authorized adapter, and record portal acknowledgment separately from local upload completion.
**Acceptance gate:** Mandatory coverage, current addenda, exact bundle hashes, and external receipt all checked.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An addendum arrives after drafting: compute its impact and invalidate only affected approvals.
- The portal limits filenames or file sizes: validate before the deadline rather than during submission.
- A certificate expires before contract start: disclose the timing and require an approved renewal commitment.
- Pricing and technical teams assume different quantities: reconcile one authoritative quantity schedule.
- A portal upload times out: inspect submission status before sending a duplicate response.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
