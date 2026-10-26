# Decomposition: Engineering change impact manager

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Trace engineering changes through parts, assemblies, stock, procedures, suppliers, and fielded products with precise applicability.

Invariant: A plausible cause is not a verified cause, local exceptions do not remove safety prerequisites, and documentary completeness does not certify physical safety.

### interpretChange

Algorithm: Compare approved design revisions and change rationale to identify functional, interface, and process effects. Separate editorial drawing updates from meaningful changes and preserve requirements that remain binding despite a local modification.

Gate: Every effect has a design-source basis and explicit scope.

Contract: ``GoalContract + PolicySelection + Relevant<EngineeringChange>[] + EvidencePage[] → StageResult<DesignDelta>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### resolveApplicability

Algorithm: Interpret effective serials, lots, dates, configurations, and authorized exceptions. Distinguish new production from retrofit obligations and preserve fielded units outside scope rather than applying the latest revision universally.

Gate: Exact applicability rules are established before downstream propagation.

Contract: ``GoalContract + PolicySelection + Relevant<DesignDelta>[] + EvidencePage[] → StageResult<ChangeApplicability>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### traceDependencies

Algorithm: Traverse bills of material, interfaces, process instructions, and supplier commitments through exact dependency graphs. Judge whether each indirect dependency is materially affected and retain uncertain impacts for engineering review instead of broad automatic replacement.

Gate: No missed critical dependency or unsupported global invalidation.

Contract: ``GoalContract + PolicySelection + Relevant<ChangeApplicability>[] + EvidencePage[] → StageResult<EngineeringImpact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planImplementation

Algorithm: Build coordinated implementation packages covering procurement, stock, production, documentation, and field actions as applicable. Sequence changes to avoid incompatible mixed configurations and identify irreversible dispositions requiring separate authorization.

Gate: Feasible version transition and explicit handling of existing stock and in-flight work.

Contract: ``GoalContract + PolicySelection + Relevant<EngineeringImpact>[] + EvidencePage[] → StageResult<ChangeWorkPackage>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### verifyConfiguration

Algorithm: Compare actual implemented configurations with approved applicability and acceptance criteria. Preserve authorized exceptions and unresolved field cases, then close the change only when required consumers and records reflect the correct revision state.

Gate: No configuration claims based solely on document publication or task closure.

Contract: ``GoalContract + PolicySelection + Relevant<ChangeWorkPackage>[] + EvidencePage[] → StageResult<ChangeVerification>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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

`reconcileUnitRetrofit.nl` advances one fielded unit only from actual installation and verification receipts for its approved serial/configuration. Mixed or unknown units remain fenced, and evidence from one unit never promotes the rest of the fleet.
