# Living evidence map — semantic decomposition

**Brief:** `applications/07-scientific-evidence-and-research-operations/031-living-evidence-map.md`
**Domain:** `Scientific evidence and research operations`

## Typed data flow

`runCase.nl` receives a case request and frozen snapshot, frames the outcome, selects policy by authority and effective scope, forms a work graph, and plans evidence. It calls the five domain stages in brief order. Each stage passes its accepted typed records to the next dependent stage; unresolved drafts remain in the proposal and do not become fabricated inputs. The root retains each full `StageResult` and all five stages' questions. Collection reconciliation, question resolution, delta assessment, review preparation, view composition, and bounded revision proposal consume those outputs before the root returns `RunProposal`.

## Domain stage ownership

### 1. `frameEligibility` → `EligibilityCriterion

**Input records:** `GoalContract`, `PolicySelection`, `Snapshot.caseRecords`, and bounded `EvidencePage[]`.
**Algorithm:** Translate the research question into explicit screening criteria and distinguish mandatory eligibility from useful descriptive metadata. Define ambiguous boundaries with examples before full screening and preserve questions requiring expert methodological judgment.
**Acceptance gate:** Criteria are frozen by version and usable on unseen records.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 2. `linkStudyReports` → `StudyIdentity

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<EligibilityCriterion>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Compare authors, enrollment windows, populations, identifiers, and methods to determine whether reports describe the same study or overlapping cohorts. Preserve uncertain links and distinguish follow-up analyses from independent replications before counting evidence.
**Acceptance gate:** Multiple reports cannot inflate the count of independent studies.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 3. `screenAndExtract` → `StudyObservation

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<StudyIdentity>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Apply eligibility to available evidence, escalating insufficient abstracts to full-text review. Extract outcomes, methods, and limitations with source locations; keep author interpretation separate from directly reported observations and exact numerical results.
**Acceptance gate:** Exclusions have criterion-specific reasons; extracted values preserve units and uncertainty.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 4. `mapClaims` → `EvidenceClaim

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<StudyObservation>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Group evidence by the actual proposition and relevant population rather than broad topic. Identify incompatible measurements and contradictory findings, qualify applicability, and expose evidence gaps without treating absent studies as proof of no effect.
**Acceptance gate:** Claims do not exceed the populations, designs, and measures of their supporting studies.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

### 5. `maintainReview` → `EvidenceUpdate

**Input records:** `GoalContract`, `PolicySelection`, `Relevant<EvidenceClaim>[]`, and bounded `EvidencePage[]`.
**Algorithm:** Process new reports, corrections, and retractions as scoped changes. Re-evaluate linked studies and dependent claims, preserve prior review snapshots, and request expert review where the new evidence changes methodological interpretation rather than merely adding another citation.
**Acceptance gate:** Retractions and corrections invalidate all affected derivatives; unchanged claims retain provenance.
**Owned children:** `prepare.nl` selects scoped evidence and missing fields; `decide.nl` makes evidence-linked atoms under the domain rule; `construct.nl` assembles the concrete record while leaving unknown values absent; `audit.nl` challenges the draft against counterevidence and calls its companion `audit/qualify.nl` for finite `pass | fail | needs-review` qualification.
**Failure behavior:** preserve the candidate as unresolved or rejected with its evidence references, coverage accounting, questions, and smallest repair target; do not weaken the gate or remove the failed item.

## Cross-cutting procedures

`frameRequest`, `selectPolicy`, `formTaskGraph`, and `planEvidence` bind the requested outcome to authorized policy, available task dependencies, and searched evidence partitions. `reconcileCollection`, `resolveQuestions`, and `assessDelta` preserve collection conflicts, all stage blockers, and scoped invalidation. `prepareReview`, `composeView`, and `proposeRevision` prepare a recipient-bound proposal, uncertainty-preserving data view, and narrow repair proposal. All outputs are included in the root result or directly consumed by the next named procedure.

## Execution boundary

Natural-language modules own domain interpretation, evidence linkage, conflict preservation, uncertainty, and review proposals. Exact host services own authorization, tenant and ACL filtering, immutable source versioning, identity joins, arithmetic and unit conversion, transactional persistence, approval-token checks, idempotent dispatch, and effect receipts. The semantic root always returns `commitAllowed: false`; no generated task trace is represented as a live-model success.

## Operational regressions

- A preprint becomes a journal article: link versions instead of counting a new study.
- A table uses different outcome units: preserve originals and require an approved conversion.
- Full text is inaccessible: record the limitation rather than infer its results from an abstract.
- A paper is retracted: invalidate dependent claims and notify review owners.
- Eligibility changes mid-review: re-screen affected records under an explicit new version.

The five operational details are separate review-required workflow scenarios. Exact gate fixtures are finite leaf judgements over the first domain gate and use one supported, one contradicted, and one evidence-incomplete world.
