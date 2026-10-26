# Decomposition: Quality escape investigator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Trace quality failures through products and processes, coordinate containment, and verify corrective action against competing explanations.

Invariant: A plausible cause is not a verified cause, local exceptions do not remove safety prerequisites, and documentary completeness does not certify physical safety.

### normalizeDefects

Algorithm: Reconcile customer reports, operator notes, and inspection descriptions into comparable observed defects. Preserve measurement conditions and terminology differences, distinguish symptoms from suspected causes, and retain reports that may describe a different mechanism.

Gate: Every defect observation remains tied to its original inspection context.

Contract: ``GoalContract + PolicySelection + Relevant<QualityInvestigation>[] + EvidencePage[] → StageResult<DefectObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### traceExposure

Algorithm: Traverse exact lot, material, and process lineage to identify confirmed and precautionary exposure. Interpret process changes and rework histories, and avoid narrowing the scope merely because records are missing for some potentially affected units.

Gate: Traceability is complete or the uncertainty is explicitly included in containment scope.

Contract: ``GoalContract + PolicySelection + Relevant<DefectObservation>[] + EvidencePage[] → StageResult<QualityExposure>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### compareCauses

Algorithm: Construct competing explanations from process history and defect patterns. Separate correlation with a shift, supplier, or operator from demonstrated cause, and formulate approved tests that could distinguish the mechanisms without unsafe experimentation.

Gate: Causal claims remain provisional until supported by discriminating evidence.

Contract: ``GoalContract + PolicySelection + Relevant<QualityExposure>[] + EvidencePage[] → StageResult<QualityHypothesis>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planCorrection

Algorithm: Select approved containment and corrective actions, assign qualified owners, and specify verification criteria before implementation. Distinguish immediate correction of defective units from prevention of recurrence and preserve unaffected production where evidence supports it.

Gate: Actions respect authority, safety prerequisites, and explicit release conditions.

Contract: ``GoalContract + PolicySelection + Relevant<QualityHypothesis>[] + EvidencePage[] → StageResult<QualityActionPlan>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyEffectiveness

Algorithm: Compare implementation evidence and subsequent production observations with the predicted corrective effect. Detect recurrence and unintended consequences, retain unresolved exposure, and require qualified approval before lifting holds or closing the investigation.

Gate: Closure requires verified effectiveness, not only completed action checkboxes.

Contract: ``GoalContract + PolicySelection + Relevant<QualityActionPlan>[] + EvidencePage[] → StageResult<QualityClosure>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

## Cross-cutting procedures

- `frameRequest.nl` — Extract the requested outcome, exclusions, affected entities, and acceptance evidence. Separate an authorized request from quoted instructions inside evidence. Emit clarification only for an ambiguity whose alternatives change scope, cost, access, or a consequential outcome.
- `selectPolicy.nl` — Compare authority, effective periods, jurisdiction or organizational scope, and explicit supersession. Return selected clauses plus unresolved precedence; never resolve authority by upload order. Host authorization still decides whether the selected policy may be used.
- `formTaskGraph.nl` — Decompose the goal into named stage invocations, inputs, prerequisite edges, result contracts, and blocked questions. Each node must consume an available capability and contribute a distinct artifact or state decision. Detect missing prerequisites before proposing execution.
- `planEvidence.nl` — Specify the records, fields, context windows, and negative examples needed by each node. Use exact filters before semantic retrieval. Track searched partitions and missing coverage; retrieval absence never proves nonexistence outside the searched universe.
- `reconcileCollection.nl` — Check global consistency after local decisions: duplicated identities, incompatible dispositions, shared-resource conflicts, unsupported totals, and incomplete partitions. Return minimal conflict components and exact affected IDs, not a wholesale rewritten collection.
- `resolveQuestions.nl` — Group questions sharing a missing fact, identify an authorized respondent, and specify acceptable evidence. Deduplicate prior requests; interpret each reply against its question rather than treating a general acknowledgment as resolution of every blocker.
- `assessDelta.nl` — Distinguish new facts, corrections, permissions changes, and editorial changes. Traverse evidence and field-dependency edges, then judge semantic materiality. Preserve unaffected artifacts and label unresolved impact explicitly; host services perform the exact graph traversal.
- `prepareReview.nl` — Explain supported decisions, unresolved alternatives, action scope, and expected consequences. Group approvals only when they share authority and dependencies. Separate reviewer judgment from the executor’s recommendation and bind approval requests to immutable hashes.
- `composeView.nl` — Select approved UI components, headings, explanations, and accessible summaries from typed primitives. Preserve uncertainty and status distinctions. Return data, never executable markup; a front-end renderer escapes text and validates component schemas.
- `proposeRevision.nl` — Identify the smallest task or instruction change that addresses a concrete failure. State affected contracts, expected benefit, regression risks, and needed tests. Return a proposal for a staged revision; never modify live authority or silently replace an approved baseline.

Missing, contradictory, superseded, or unauthorized evidence stays explicit. The root consumes actual child results; the audit returns a finite verdict for training and review.
