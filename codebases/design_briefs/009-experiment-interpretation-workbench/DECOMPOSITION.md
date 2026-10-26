# Experiment interpretation workbench — semantic decomposition

**Brief:** `applications/02-customer-research-and-product-intelligence/009-experiment-interpretation-workbench.md`
**Domain:** `Customer research and product intelligence`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverDesign` → `ExperimentDesign

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read the original plan and subsequent amendments in temporal order. Distinguish preregistered criteria from retrospective choices, identify the assignment and observation units, and preserve deviations that could change interpretation even when the experiment ran successfully.
**Acceptance gate:** Every design claim links to its effective plan version.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `auditMeasurement` → `MeasurementFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ExperimentDesign>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare metric definitions, event schemas, instrumentation notes, and missingness reports. Identify whether treatment changed the measurement process itself; formulate exact diagnostic queries instead of assuming a numerical difference reflects changed customer behavior.
**Acceptance gate:** Incomparable measurement windows cannot enter an unqualified pooled result.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `interpretResults` → `ResultClaim

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MeasurementFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Consume approved statistical outputs rather than calculating inference through prose. Relate results to the specified estimand, inspect multiplicity and subgroup selection, and distinguish evidence of little effect from an underpowered or compromised experiment.
**Acceptance gate:** Claimed population, direction, and uncertainty match the actual computed analysis.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `reconcileFeedback` → `MechanismFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ResultClaim>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare qualitative accounts with result patterns to formulate plausible mechanisms, not causal proof. Seek dissatisfied successes and satisfied failures that challenge the simple explanation, and identify evidence capable of distinguishing rival accounts.
**Acceptance gate:** Anecdotes cannot override valid numerical results or establish unmeasured causality.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `compileFollowups` → `AnalysisOrder

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MechanismFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Convert unresolved issues into typed analysis plans with cohorts, time windows, allowed operations, and interpretation rules. Separate confirmatory follow-up from exploratory searching and require a fresh data or holdout plan where the proposed claim needs one.
**Acceptance gate:** Work orders compile against available schemas and expose every post-hoc choice.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An event name changes halfway through: bind metrics to schema versions before aggregation.
- A subgroup looks exceptional after many searches: label it exploratory and preserve the search history.
- Assignment is by team but rows are users: reject analyses with the wrong independence unit.
- A result arrives after the decision meeting: version the recommendation rather than silently editing minutes.
- A metric has no trustworthy denominator: withhold the rate and request exposure reconstruction.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
