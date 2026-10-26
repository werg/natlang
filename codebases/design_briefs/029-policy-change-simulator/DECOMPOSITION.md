# Policy change simulator — semantic decomposition

**Brief:** `applications/06-contracts-policies-and-information-governance/029-policy-change-simulator.md`
**Domain:** `Contracts, policies, and information governance`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `compilePolicyDifference` → `PolicyDelta

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Compare baseline and proposed policies at the level of conditions, exceptions, authority, and consequences. Separate editorial changes from changed behavior and identify ambiguous provisions that prevent a determinate comparison.
**Acceptance gate:** Every behavioral delta has source support and preserved safeguards are explicit.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `prepareCases` → `PolicyCaseFacts

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PolicyDelta>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Normalize historical or synthetic cases into evidence-backed facts without adding missing information. Preserve uncertainty, policy-relevant context, and data-access restrictions; do not infer sensitive attributes merely to complete a simulation.
**Acceptance gate:** Case facts remain identical across baseline and candidate runs.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `compareOutcomes` → `OutcomeDifference

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<PolicyCaseFacts>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Execute both policy versions with isolated state and compare duties, permissions, timing, and escalation. Trace differences to specific changed conditions and separate genuine policy effects from nondeterministic or unsupported semantic judgments.
**Acceptance gate:** Paired runs share facts and executor settings; unexplained differences remain diagnostic failures.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `generateBoundaries` → `BoundaryScenario

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<OutcomeDifference>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct near-neighbor cases around changed thresholds, exceptions, and overlapping categories. Vary one consequential fact at a time, include adverse interactions, and identify scenarios whose outcome exposes an ambiguity requiring a policy decision.
**Acceptance gate:** Perturbation is controlled and does not smuggle in multiple explanatory changes.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `preparePolicyOptions` → `PolicyOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<BoundaryScenario>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Summarize who and what changes under each candidate, then propose clarifications or alternatives for the policy owner. Preserve minority and exceptional cases, report coverage limits, and avoid choosing normative priorities not supplied by the organization.
**Acceptance gate:** Policy owner sees exact changed-case sets and unresolved judgments before adoption.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- Historical cases lack a new required field: report indeterminate outcomes rather than imputing it.
- A policy owner edits the candidate mid-run: fork the simulation snapshot.
- A rare exception disappears: preserve its regression case even if aggregate scores improve.
- Outcome counts change due to deduplication: separate data changes from policy effects.
- A policy is adopted: retain the tested version and explicitly revalidate subsequent edits.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
