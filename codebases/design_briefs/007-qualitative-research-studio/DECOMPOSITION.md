# Qualitative research studio — semantic decomposition

**Brief:** `applications/02-customer-research-and-product-intelligence/007-qualitative-research-studio.md`
**Domain:** `Customer research and product intelligence`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `frameInquiry` → `InquiryFrame

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Interpret the decision that research should inform and decompose it into answerable questions. Identify plausible rival explanations and what the available corpus cannot establish; avoid converting an exploratory question into a predetermined conclusion.
**Acceptance gate:** Every subquestion has observable evidence and an explicit inferential limit.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `selectSample` → `SamplingPlan

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<InquiryFrame>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Define strata from the inquiry, not from convenient examples, then request exact selection within each stratum. Include negative and atypical cases, record nonresponse, and freeze a sampling manifest before coding the main set.
**Acceptance gate:** Manifest preserves inclusion probabilities where applicable and explains purposive selection elsewhere.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `buildCodebook` → `CodeDefinition

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<SamplingPlan>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Draft codes from pilot material and research concepts, specifying boundaries through positive and negative examples. Separate descriptive labels from inferred explanations; reconcile overlapping codes without forcing genuinely multidimensional evidence into one category.
**Acceptance gate:** Each code is usable on unseen examples and has a discriminating exclusion rule.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `codeAndReconcile` → `CodingDecision

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CodeDefinition>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Code bounded passages with their necessary context, then compare independent assessments on a sampled overlap. Investigate disagreements as possible codebook defects; when a definition changes, mark all affected earlier decisions stale before recoding.
**Acceptance gate:** Code-version consistency and no silent replacement of disagreement with majority confidence.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `synthesizeFindings` → `ResearchFinding

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<CodingDecision>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Build findings around supported mechanisms and recurring patterns, actively search for counterexamples, and distinguish an absent observation from evidence of absence. Return unresolved questions and a traceable path from each conclusion to the coded passages.
**Acceptance gate:** No causal or population-level claim beyond the study design; counterevidence cannot be omitted.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A participant switches language mid-interview: retain original passages alongside reviewed translations.
- A code changes after half the corpus is processed: recompute affected records, not only future ones.
- A quotation may identify its speaker indirectly: preview re-identification risk before export.
- Two researchers disagree productively: preserve minority interpretation in the synthesis packet.
- The study stops early: publish achieved coverage and unmet stopping conditions.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
