# Cross-study measurement harmonizer — semantic decomposition

**Brief:** `applications/07-scientific-evidence-and-research-operations/032-cross-study-measurement-harmonizer.md`
**Domain:** `Scientific evidence and research operations`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `recoverMeasures` → `MeasureDefinition

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Read dictionaries, questionnaires, protocols, and usage notes together to recover what each variable measures. Distinguish similar labels from equivalent constructs and record timing, respondent, and collection-method differences that could change interpretation.
**Acceptance gate:** No mapping based only on column name or numeric range.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `assessComparability` → `MeasureRelation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MeasureDefinition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare constructs, scales, recall periods, and populations against the intended analysis. Separate exact unit conversion from inferential harmonization, and identify subsets where comparability holds without claiming it for the entire dataset.
**Acceptance gate:** Every accepted relation specifies the assumptions and applicable subset.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `proposeTransforms` → `HarmonizationRule

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<MeasureRelation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Select documented transformations or category mappings and describe lost distinctions. Delegate arithmetic to exact services, refuse unsupported imputation, and preserve source variables so analysts can inspect or reverse the transformation where possible.
**Acceptance gate:** Transformation is defined for every accepted input state, including missing and out-of-range values.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `constructCohort` → `HarmonizedSlice

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<HarmonizationRule>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Define comparable cohorts before materialization, apply approved transformations in a sandbox, and reconcile row counts and exclusions. Keep missingness reasons and study membership available rather than flattening them into one generic null category.
**Acceptance gate:** Exact row lineage, unit consistency, and no unsupported cross-study comparability.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `auditSensitivity` → `HarmonizationAudit

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<HarmonizedSlice>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare reasonable alternative mappings and cohort restrictions to reveal dependence on harmonization choices. Produce an analyst-facing report distinguishing stable findings from those driven by assumptions and identify measurements that must remain separate.
**Acceptance gate:** Sensitivity alternatives are justified and do not selectively hide inconvenient results.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A zero is a valid response in one study but missing in another: preserve distinct missingness semantics.
- A category combines two concepts: do not invent a finer split without evidence.
- A unit changes within a dataset: bind transformations to row-level collection versions.
- Participant overlap exists across studies: retain linkage uncertainty for downstream analysis.
- A mapping is revised: invalidate only derived slices using that rule.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
