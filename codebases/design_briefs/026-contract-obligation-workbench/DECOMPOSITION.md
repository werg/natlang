# Contract obligation workbench — semantic decomposition

**Brief:** `applications/06-contracts-policies-and-information-governance/026-contract-obligation-workbench.md`
**Domain:** `Contracts, policies, and information governance`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `parseObligations` → `ContractObligation

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Extract atomic obligations with actor, beneficiary, trigger, deadline, and exceptions. Separate descriptions and aspirations from operative provisions, retain cross-references, and preserve unknown roles rather than assigning responsibility from nearby names.
**Acceptance gate:** Each obligation retains all material conditions and supporting clause spans.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `resolveInteractions` → `ObligationRelation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ContractObligation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret cross-references and interactions among provisions under the supplied authority hierarchy. Distinguish a narrow exception from replacement of an entire obligation and preserve unresolved conflicts for qualified review instead of guessing a priority rule.
**Acceptance gate:** Every relation has scope and source support; conflict is not silently converted into supersession.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `assessEvents` → `TriggerFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ObligationRelation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare incoming operational events with contractual triggers and conditions. Distinguish an allegation that a trigger occurred from established evidence, and request the smallest missing fact when activation would materially change a duty or deadline.
**Acceptance gate:** Trigger activation requires evidence under the selected interpretation version.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `planPerformance` → `ObligationTask

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<TriggerFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Turn active obligations into bounded tasks with ownership, dependencies, and exact computed dates. Preserve legal interpretation separately from operational scheduling and identify impossible or conflicting task requirements before issuing commitments.
**Acceptance gate:** No unowned active obligation; dates use the configured calendar and explicit deadline rule.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `verifyDischarge` → `ObligationStatus

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ObligationTask>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare completed actions with the actual duty and any acceptance condition. Distinguish performance from waiver or settlement, record reviewer-approved interpretations, and keep disputed discharge visible rather than closing on an internal completion flag.
**Acceptance gate:** Satisfaction is supported; waivers require their own scoped authority evidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A deadline uses business days: store the agreed calendar rather than assuming local weekends.
- A party changes its legal name: preserve identity and do not treat it as a new counterparty automatically.
- A notice is sent to the wrong address: separate dispatch from contractually effective notice.
- A clause is disputed: display alternative operational consequences pending review.
- A confidential clause supports a shared task: expose only the minimum necessary instruction.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
