# Negative-result learning archive — semantic decomposition

**Brief:** `applications/07-scientific-evidence-and-research-operations/035-negative-result-learning-archive.md`
**Domain:** `Scientific evidence and research operations`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `reconstructAttempt` → `ResearchAttempt

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read plans, execution logs, and researcher notes to reconstruct the attempted test. Separate intended from actual procedure and identify incomplete steps, measurement failures, or deviations that limit what the result can establish.
**Acceptance gate:** The archive records what actually happened, not only what was planned.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `classifyFailure` → `NegativeFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<ResearchAttempt>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Determine whether evidence supports a negative result, an invalid measurement, an unfinished procedure, or genuine uncertainty. Preserve the distinction even when all cases were informally described as failures and avoid stronger conclusions than the attempt warrants.
**Acceptance gate:** A failed process cannot be treated as evidence against the hypothesis it never validly tested.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `compareExplanations` → `FailureExplanation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<NegativeFinding>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Construct competing explanations grounded in observed deviations and results. Separate demonstrated causes from plausible hypotheses, seek contradictory observations, and identify shared mechanisms across attempts without assuming all similar outcomes have one cause.
**Acceptance gate:** Each causal claim has evidence or an explicit hypothetical status.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `designFollowups` → `DiscriminatingExperiment

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<FailureExplanation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Propose bounded low-risk follow-ups that distinguish rival explanations with minimal changes. Specify what each outcome would mean before execution and avoid repeating the same invalid procedure or adding so many changes that the result becomes uninterpretable.
**Acceptance gate:** Follow-up has a discriminating prediction and preserves approved safety and resource constraints.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `maintainLessons` → `ResearchLesson

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<DiscriminatingExperiment>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Publish a scoped lesson with conditions under which it applies and links to contradictory or superseding evidence. Update recommendations when new attempts arrive, preserving the historical knowledge state and uncertainty rather than rewriting failures into hindsight certainty.
**Acceptance gate:** Lessons do not generalize beyond tested conditions or erase negative evidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- An experiment was abandoned before measurement: label it incomplete rather than negative.
- A later success uses different conditions: narrow the earlier lesson instead of deleting it.
- A researcher fears reputational exposure: support restricted attribution with authorized methodological access.
- Several failures share one broken instrument: link the common process issue without multiplying scientific evidence.
- A follow-up changes three variables: flag its weak discriminating power before execution.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
