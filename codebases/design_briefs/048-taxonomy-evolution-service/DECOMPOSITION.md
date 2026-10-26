# Decomposition: Taxonomy evolution service

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Evolve classifications as new concepts appear while preserving meaningful boundaries, stable identities, and historical comparability.

Invariant: A matching label is not semantic equivalence, a missing value may carry meaning, and exact aggregation is correct only after the intended population and predicates are established.

### assessFit

Algorithm: Compare incoming records with category definitions and examples, not names alone. Distinguish a genuinely new concept from a poorly worded existing category, and retain ambiguous membership where the taxonomy permits multiple dimensions.

Gate: Classification uses the current definition version and preserves uncertainty.

Contract: ``GoalContract + PolicySelection + Relevant<TaxonomyChange>[] + EvidencePage[] → StageResult<CategoryFit>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### diagnoseStructure

Algorithm: Analyze repeated fit problems to identify structural defects. Separate insufficient annotation guidance from real conceptual change, inspect negative examples, and avoid proliferating categories for every rare wording variant.

Gate: Proposed structural changes are supported by multiple informative examples or a decisive new requirement.

Contract: ``GoalContract + PolicySelection + Relevant<CategoryFit>[] + EvidencePage[] → StageResult<TaxonomyDefect>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### designCategoryRevision

Algorithm: Draft revised definitions, inclusion and exclusion rules, and parent-child relationships. Preserve stable category IDs where meaning remains stable, create new identities when meaning changes, and explicitly map split or merged historical categories.

Gate: No cycle, contradictory boundary, or silent reuse of an ID for a different concept.

Contract: ``GoalContract + PolicySelection + Relevant<TaxonomyDefect>[] + EvidencePage[] → StageResult<CategoryRevision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### simulateReclassification

Algorithm: Reclassify only potentially affected records under baseline and candidate definitions, compare changes, and inspect boundary cases. Build explicit bridges for historical reporting and retain cases that cannot be mapped uniquely after a split.

Gate: Exact changed-record set and no fabricated historical precision.

Contract: ``GoalContract + PolicySelection + Relevant<CategoryRevision>[] + EvidencePage[] → StageResult<TaxonomyImpact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### promoteTaxonomy

Algorithm: Present impact, examples, and reporting consequences to the steward, then publish a versioned release and migration plan. Notify dependent systems, invalidate only relevant classification caches, and preserve old definitions for historical queries.

Gate: Approved definitions, migration scope, and dependent consumer compatibility are verified.

Contract: ``GoalContract + PolicySelection + Relevant<TaxonomyImpact>[] + EvidencePage[] → StageResult<TaxonomyRelease>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
