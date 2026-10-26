# Decomposition: Adaptive reuse feasibility studio

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Generate and compare building reuse concepts grounded in supplied constraints, stakeholder goals, and explicit unknown site conditions.

Invariant: A site instruction may not authorize a commercial change, similar bid totals may cover different scope, and a complete dossier does not prove the building is technically safe.

### recoverSiteFacts

Algorithm: Reconcile inventories, surveys, historical plans, and stakeholder descriptions into verified facts and unknowns. Distinguish current observations from outdated drawings and preserve uncertainties that could materially change feasibility or cost.

Gate: No assumed structural or regulatory fact without supplied evidence.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseStudy>[] + EvidencePage[] → StageResult<ReuseSiteFact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretGoals

Algorithm: Translate stakeholder objectives into required functions and explicit tradeoffs. Separate mandatory supplied constraints from preferences, identify incompatible goals, and avoid choosing political or commercial priorities not authorized by the project brief.

Gate: Goals and constraints retain accountable owners and authority basis.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseSiteFact>[] + EvidencePage[] → StageResult<ReuseGoal>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### generateConcepts

Algorithm: Invent distinct reuse concepts that exploit the building’s known characteristics while respecting supplied constraints. Describe functional organization and phasing, preserve uncertainty about unverified conditions, and avoid presenting conceptual layouts as certified engineering designs.

Gate: Concepts satisfy known hard constraints and expose every consequential assumption.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseGoal>[] + EvidencePage[] → StageResult<ReuseConcept>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planInvestigations

Algorithm: Identify the surveys, specialist reviews, or evidence needed to distinguish feasible concepts. Prioritize investigations by their ability to change a decision, and specify how positive, negative, or inconclusive findings would alter the concept.

Gate: Each investigation resolves a named uncertainty and uses qualified professionals where required.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseConcept>[] + EvidencePage[] → StageResult<ReuseInvestigation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### compareFeasibility

Algorithm: Compare concepts using exact supplied estimates, known constraints, functional benefits, and unresolved conditions. Present phased or contingent options and clear next decisions, separating creative promise from verified technical or regulatory feasibility.

Gate: No unqualified feasibility or compliance claim beyond available professional evidence.

Contract: ``GoalContract + PolicySelection + Relevant<ReuseInvestigation>[] + EvidencePage[] → StageResult<ReuseOptionReview>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
