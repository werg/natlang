# Decomposition: Citizen-science observation curator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Curate field observations with careful terminology, duplication, metadata, and uncertainty handling while preserving legitimate rare reports.

Invariant: An unusual observation is not necessarily an error, an intervention’s ecological effect is uncertain, and activity completion does not prove environmental impact.

### normalizeObservations

Algorithm: Parse field notes and structured submissions while preserving original wording, location precision, and observation method. Distinguish what was directly seen from inferred identity or cause, and flag missing metadata without inventing exact coordinates or times.

Gate: Original evidence and uncertainty remain recoverable.

Contract: ``GoalContract + PolicySelection + Relevant<ObservationReview>[] + EvidencePage[] → StageResult<FieldObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### resolveTerminology

Algorithm: Interpret local names and descriptive phrases against project terminology and context. Retain multiple plausible identities when evidence is insufficient, and avoid forcing unusual observations into familiar categories merely to complete the dataset.

Gate: Concept assignments have discriminating evidence or remain provisional.

Contract: ``GoalContract + PolicySelection + Relevant<FieldObservation>[] + EvidencePage[] → StageResult<ObservationInterpretation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### detectDuplicates

Algorithm: Compare observer, time, location, media hashes, and narrative context to identify repeated submissions of one event. Preserve independent observations of the same phenomenon and avoid merging nearby events solely because they share a likely identity.

Gate: Event counts distinguish duplicate submissions from independent corroboration.

Contract: ``GoalContract + PolicySelection + Relevant<ObservationInterpretation>[] + EvidencePage[] → StageResult<ObservationLink>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### requestVerification

Algorithm: Form targeted follow-up requests that could distinguish plausible interpretations or correct metadata. Use respectful wording, avoid leading the observer toward the expected label, and route specialist-sensitive cases for expert review.

Gate: Questions are informative, feasible, and do not request unsafe field activity.

Contract: ``GoalContract + PolicySelection + Relevant<ObservationLink>[] + EvidencePage[] → StageResult<ObservationQuestion>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### publishCurated

Algorithm: Prepare a curated record with confidence categories grounded in evidence, preserved alternatives, and project-specific disclosure controls. Publish provisional and verified statuses distinctly and update derivatives when experts or observers provide corrections.

Gate: Rare but supported observations are retained; sensitive locations are appropriately generalized.

Contract: ``GoalContract + PolicySelection + Relevant<ObservationQuestion>[] + EvidencePage[] → StageResult<CuratedObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
