# Decomposition: Learning portfolio evidence assessor

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Assess competency evidence across artifacts and formulate targeted next tasks without confusing self-description with demonstrated ability.

Invariant: A wrong answer may reflect ambiguity rather than misconception, fluent explanation is not demonstrated competence, and accommodations must not be inferred from sensitive attributes.

### indexEvidence

Algorithm: Read artifacts, reflections, and supervisor comments to identify observable work and its context. Separate the learner’s own contribution from group output, distinguish direct evidence from self-report, and preserve uncertainty about authorship or assistance.

Gate: Evidence type and authorship status are explicit.

Contract: ``GoalContract + PolicySelection + Relevant<PortfolioReview>[] + EvidencePage[] → StageResult<PortfolioEvidence>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### mapCompetencies

Algorithm: Compare evidence with the actual competency definition and performance level. Recognize alternative demonstrations, avoid matching by keywords alone, and identify which facets are supported versus merely mentioned in a reflection.

Gate: Every mapping names the demonstrated behavior and source location.

Contract: ``GoalContract + PolicySelection + Relevant<PortfolioEvidence>[] + EvidencePage[] → StageResult<CompetencyMapping>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### assessSufficiency

Algorithm: Evaluate breadth, recency, independence, and consistency under the supplied assessment standard. Prevent double-counting one artifact as multiple independent demonstrations and preserve contradictory evidence rather than averaging it away.

Gate: Sufficiency follows the approved standard and does not rely on unsupported confidence.

Contract: ``GoalContract + PolicySelection + Relevant<CompetencyMapping>[] + EvidencePage[] → StageResult<CompetencyFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### formNextTasks

Algorithm: Design the smallest authentic task that fills a consequential evidence gap. Tailor context and resources to the learner while preserving the competency, and specify what would count as success before the new work is submitted.

Gate: Task directly addresses the missing facet and is feasible under stated constraints.

Contract: ``GoalContract + PolicySelection + Relevant<CompetencyFinding>[] + EvidencePage[] → StageResult<EvidenceTask>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### prepareAssessment

Algorithm: Assemble a reviewable recommendation with evidence, gaps, and proposed next work. Allow the learner to correct factual attribution, preserve assessor disagreements, and record the authorized decision separately from the program’s recommendation.

Gate: High-stakes outcomes require accountable assessor approval and an appeal path.

Contract: ``GoalContract + PolicySelection + Relevant<EvidenceTask>[] + EvidencePage[] → StageResult<PortfolioDisposition>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
