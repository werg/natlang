# Decomposition: Custom fabrication quoting desk

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Turn informal fabrication requests into manufacturable alternatives, explicit assumptions, exact quotes, and coherent production plans.

Invariant: A plausible cause is not a verified cause, local exceptions do not remove safety prerequisites, and documentary completeness does not certify physical safety.

### recoverRequirements

Algorithm: Interpret sketches, descriptions, and intended use to distinguish functional requirements from aesthetic preferences and unstated assumptions. Identify consequential missing dimensions or loads and request clarification rather than inventing specifications that determine manufacturability or safety.

Gate: Critical requirements are confirmed or explicitly block a firm quote.

Contract: ``GoalContract + PolicySelection + Relevant<FabricationQuote>[] + EvidencePage[] → StageResult<FabricationRequirement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generateConcepts

Algorithm: Propose distinct concepts using the shop’s supplied capabilities and approved materials. Explain appearance, cost, durability, and lead-time tradeoffs without claiming unsupported performance; preserve essential customer constraints across creative alternatives.

Gate: Concept is feasible under documented shop capability and approved design constraints.

Contract: ``GoalContract + PolicySelection + Relevant<FabricationRequirement>[] + EvidencePage[] → StageResult<FabricationConcept>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planProduction

Algorithm: Translate each concept into a production route with material preparation, operations, finishing, assembly, and inspection. Identify outsourced steps and dependencies, reserve realistic capacity, and retain specialist approval requirements for consequential structural or safety questions.

Gate: Exact resource and process feasibility; no unsupported engineering certification.

Contract: ``GoalContract + PolicySelection + Relevant<FabricationConcept>[] + EvidencePage[] → StageResult<FabricationRoute>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### calculateQuote

Algorithm: Use exact costing services with documented quantities, rates, waste assumptions, and supplier quotes. Distinguish fixed price from allowance or estimate, expose unknowns, and prevent a creative design change from silently invalidating the cost basis.

Gate: Quote arithmetic and scope match the proposed route and selected concept.

Contract: ``GoalContract + PolicySelection + Relevant<FabricationRoute>[] + EvidencePage[] → StageResult<FabricationEstimate>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### confirmScope

Algorithm: Present drawings or structured specifications, finish samples, assumptions, and acceptance criteria for customer confirmation. Bind the quote to the accepted revision, then create production work only after material scope and commercial terms are approved.

Gate: Quoted object, accepted specification, and production route are mutually consistent.

Contract: ``GoalContract + PolicySelection + Relevant<FabricationEstimate>[] + EvidencePage[] → StageResult<FabricationAgreement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
