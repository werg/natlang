# Decomposition: Semantic analysis work-order compiler

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Compile a broad natural-language analysis request into a typed, auditable plan that combines semantic judgments with exact data operations.

Invariant: A matching label is not semantic equivalence, a missing value may carry meaning, and exact aggregation is correct only after the intended population and predicates are established.

### formalizeQuestion

Algorithm: Parse the request into population, entities, semantic predicates, temporal relations, grouping, and desired output. Generate competing interpretations only where they change the answer; resolve decisive ambiguity before choosing a convenient available column.

Gate: Every phrase affecting inclusion, timing, or aggregation maps to an explicit meaning component.

Contract: ``GoalContract + PolicySelection + Relevant<AnalysisRequest>[] + EvidencePage[] → StageResult<AnalysisMeaning>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### inspectData

Algorithm: Inspect schemas, dictionaries, representative records, and access constraints to determine whether the question is answerable. Distinguish absent data from inaccessible data and identify proxy fields that would change the meaning rather than silently substituting them.

Gate: Each required concept has a supported mapping or an explicit unanswerable component.

Contract: ``GoalContract + PolicySelection + Relevant<AnalysisMeaning>[] + EvidencePage[] → StageResult<DataSuitability>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### compileGraph

Algorithm: Construct a typed DAG of exact filters, joins, bounded semantic classification, temporal matching, and aggregation. Push exact restrictive filters before semantic work when meaning is preserved; attach provenance and missing-data rules to every edge and reject unsupported operations.

Gate: Graph typechecks, has no cycle, and preserves the formalized population and output grain.

Contract: ``GoalContract + PolicySelection + Relevant<DataSuitability>[] + EvidencePage[] → StageResult<AnalysisWorkGraph>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### executeAndAudit

Algorithm: Execute finite partitions, cache semantic judgments by record and rubric version, and reconcile every input disposition. Audit sampled positive and negative labels, exact join cardinalities, temporal boundaries, and denominators before materializing the final aggregate.

Gate: No silent row loss, duplicate join inflation, unclassified inclusion, or unsupported denominator.

Contract: ``GoalContract + PolicySelection + Relevant<AnalysisWorkGraph>[] + EvidencePage[] → StageResult<AnalysisSlice>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### explainResult

Algorithm: Formulate the result using exact computed values and the approved question interpretation. Explain exclusions, uncertainty, and unanswerable components; attach the executable work order and lineage so another analyst can reproduce or revise the analysis.

Gate: Every numerical claim matches the materialized result and every semantic qualification matches the plan.

Contract: ``GoalContract + PolicySelection + Relevant<AnalysisSlice>[] + EvidencePage[] → StageResult<AnalysisAnswer>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
