# Decomposition: Change-order impact coordinator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Determine the scope and downstream consequences of site changes and prepare evidence-backed change packages for review.

Invariant: A site instruction may not authorize a commercial change, similar bid totals may cover different scope, and a complete dossier does not prove the building is technically safe.

### reconstructInstruction

Algorithm: Reconcile site notes, messages, and drawing annotations to identify the actual requested action and its issuer. Distinguish clarification, proposed change, and authorized instruction; preserve conflicting versions and do not infer commercial authority from site presence.

Gate: Instruction scope and authority are separately evidenced.

Contract: ``GoalContract + PolicySelection + Relevant<ChangeOrderCase>[] + EvidencePage[] → StageResult<SiteInstruction>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### compareScope

Algorithm: Compare the instruction with the agreed baseline and approved prior changes. Identify additions, omissions, substitutions, or clarifications, preserve exclusions, and avoid treating a contractor’s preferred method as contractual scope unless the baseline supports it.

Gate: Each claimed change maps to a specific baseline requirement or omission.

Contract: ``GoalContract + PolicySelection + Relevant<SiteInstruction>[] + EvidencePage[] → StageResult<ScopeDifference>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### traceDependencies

Algorithm: Trace design, procurement, trade interfaces, and schedule dependencies using exact project graphs. Judge which dependencies are materially affected, distinguish potential from confirmed delay, and identify evidence needed to support commercial or timing claims.

Gate: Claimed downstream effects have dependency paths and explicit assumptions.

Contract: ``GoalContract + PolicySelection + Relevant<ScopeDifference>[] + EvidencePage[] → StageResult<ConstructionImpact>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### formOptions

Algorithm: Construct feasible alternatives such as resequencing, substitution, or phased completion within approved design constraints. Obtain exact estimates from authorized services, disclose assumptions, and preserve professional review requirements for structural or safety consequences.

Gate: Option is technically reviewable, commercially explicit, and dependency-feasible.

Contract: ``GoalContract + PolicySelection + Relevant<ConstructionImpact>[] + EvidencePage[] → StageResult<ChangeOption>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### assembleChange

Algorithm: Prepare a coordinated scope, cost, timing, and responsibility package with the relevant evidence. Separate agreed changes from disputed claims, bind approvals to the exact revision, and update downstream plans only after the appropriate authorities accept the change.

Gate: No unapproved scope or schedule change silently enters the committed baseline.

Contract: ``GoalContract + PolicySelection + Relevant<ChangeOption>[] + EvidencePage[] → StageResult<ChangePackage>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
