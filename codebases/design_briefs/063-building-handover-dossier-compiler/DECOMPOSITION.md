# Decomposition: Building handover dossier compiler

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Assemble a traceable building handover with asset evidence, outstanding defects, training acknowledgments, and explicit readiness gaps.

Invariant: A site instruction may not authorize a commercial change, similar bid totals may cover different scope, and a complete dossier does not prove the building is technically safe.

### recoverRequirements

Algorithm: Read the supplied handover specification, asset schedules, and contractual requirements to identify required documents, tests, training, and approvals. Separate documentary requirements from technical sign-off and preserve conditions specific to asset classes or building areas.

Gate: Every required handover item has an explicit source and acceptance condition.

Contract: ``GoalContract + PolicySelection + Relevant<BuildingHandover>[] + EvidencePage[] → StageResult<HandoverRequirement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### resolveAssets

Algorithm: Reconcile asset tags, drawing labels, manuals, and contractor naming conventions. Distinguish identical models at different locations, preserve serial and revision data, and flag ambiguous identity before attaching a certificate or manual to the wrong asset.

Gate: Evidence attaches to the correct asset instance and revision.

Contract: ``GoalContract + PolicySelection + Relevant<HandoverRequirement>[] + EvidencePage[] → StageResult<AssetIdentity>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### matchArtifacts

Algorithm: Compare submitted manuals, inspection records, certificates, and training logs with the actual requirement. Distinguish a blank form from completed evidence and a generic manual from the installed configuration; retain missing signatures or scope mismatches.

Gate: Artifact presence alone does not satisfy substantive evidence requirements.

Contract: ``GoalContract + PolicySelection + Relevant<AssetIdentity>[] + EvidencePage[] → StageResult<HandoverEvidence>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planClosure

Algorithm: Group gaps into actionable requests for contractors, inspectors, or facilities staff. Prioritize dependencies affecting safe or practical use under supplied rules, and preserve accepted temporary arrangements with owners and expiry rather than hiding them as complete.

Gate: Every mandatory gap has an accountable closure path or explicit escalation.

Contract: ``GoalContract + PolicySelection + Relevant<HandoverEvidence>[] + EvidencePage[] → StageResult<HandoverClosureTask>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### assembleDossier

Algorithm: Produce a navigable asset-linked dossier and a separate outstanding-obligations register. Verify search, links, file integrity, and recipient access, then record documentary acceptance independently from any qualified technical approval or occupancy decision.

Gate: No claim of technical readiness based solely on a complete document bundle.

Contract: ``GoalContract + PolicySelection + Relevant<HandoverClosureTask>[] + EvidencePage[] → StageResult<HandoverDossier>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
