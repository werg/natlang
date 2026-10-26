# Research resource and experiment rescue coordinator — semantic decomposition

**Brief:** `applications/07-scientific-evidence-and-research-operations/034-research-resource-and-experiment-rescue-coordinator.md`
**Domain:** `Scientific evidence and research operations`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `interpretRequirements` → `ExperimentRequirement

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read approved protocols and researcher requests to distinguish essential experimental conditions from convenient preferences. Identify sample timing, equipment capabilities, and required personnel without changing scientific procedures or inferring unapproved substitutions.
**Acceptance gate:** Essential conditions are source-backed and unknown compatibility remains blocked.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `modelResources` → `ResourceCapability

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExperimentRequirement>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Reconcile calendars, instrument status, maintenance records, and operator qualifications. Distinguish booked availability from actual readiness and identify resources that appear interchangeable but differ in capability or validated use.
**Acceptance gate:** Exact capacity and qualification checks precede scheduling.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `constructSchedule` → `ExperimentSlot

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ResourceCapability>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Submit hard resource and timing constraints to an exact scheduler, then evaluate feasible schedules against supplied priorities and scientific dependencies. Keep optional preferences visible and explain unavoidable conflicts instead of silently dropping experiments.
**Acceptance gate:** No double booking, expired sample window, or missing required operator.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `planRescue` → `RescueOption

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExperimentSlot>[]`, and bounded `EvidencePage[]`.
**Algorithm:** When a resource or sample becomes unavailable, generate alternatives such as resequencing, qualified substitution, or postponement. Explicitly state scientific deviations and seek investigator approval where the original protocol would no longer hold.
**Acceptance gate:** Recovery preserves essential conditions or records an approved protocol deviation.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `coordinateReadiness` → `ReadinessCheck

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<RescueOption>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Collect readiness confirmations from resource owners and investigators, verify current status immediately before execution, and issue role-specific instructions. Preserve cancellations and rescheduling history so downstream analyses know which actual conditions applied.
**Acceptance gate:** A scheduled slot is not a go decision; all required readiness evidence must be current.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An instrument is available but calibration has expired: block readiness without deleting its booking.
- A sample’s usable window shrinks: prioritize explicit protocol constraints rather than calendar convenience.
- Two projects share an operator: schedule the person as a constrained resource.
- A substitute instrument changes measurement behavior: require investigator review.
- A cancellation occurs after preparation: record consumed materials and downstream consequences.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
