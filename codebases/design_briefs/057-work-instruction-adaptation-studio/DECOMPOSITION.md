# Decomposition: Work instruction adaptation studio

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Adapt approved master procedures to local equipment and contexts while preserving prerequisites, safeguards, and traceable deviations.

Invariant: A plausible cause is not a verified cause, local exceptions do not remove safety prerequisites, and documentary completeness does not certify physical safety.

### recoverProcedure

Algorithm: Parse the master instruction into actions, guards, prerequisites, and verification obligations. Distinguish explanatory advice from mandatory controls, preserve warnings in their operative context, and identify cross-references needed to execute the procedure correctly.

Gate: Every mandatory control has a traceable source and cannot disappear during simplification.

Contract: ``GoalContract + PolicySelection + Relevant<InstructionAdaptation>[] + EvidencePage[] → StageResult<ProcedureRequirement>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### mapLocalContext

Algorithm: Compare local equipment, materials, staffing, and terminology with the master procedure. Separate harmless naming differences from capability changes and identify where local conditions make a step impossible or require engineering review.

Gate: Local compatibility is verified; unknown critical differences block adaptation.

Contract: ``GoalContract + PolicySelection + Relevant<ProcedureRequirement>[] + EvidencePage[] → StageResult<LocalProcedureConstraint>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructVariant

Algorithm: Rewrite steps for the local context using approved terminology and resources. Preserve sequence, guards, stop conditions, and verification; represent genuine deviations explicitly rather than hiding them in friendlier wording or inferred substitutions.

Gate: All master safety and prerequisite obligations survive unless explicitly superseded by authorized engineering change.

Contract: ``GoalContract + PolicySelection + Relevant<LocalProcedureConstraint>[] + EvidencePage[] → StageResult<InstructionVariant>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### replayProcedure

Algorithm: Generate ordinary and adverse scenarios involving local exceptions, missing resources, and interrupted work. Replay the adapted procedure to detect ambiguous instructions, skipped prerequisites, or branches that cannot reach a safe terminal state.

Gate: No scenario bypasses a retained safeguard or requires an unavailable capability.

Contract: ``GoalContract + PolicySelection + Relevant<InstructionVariant>[] + EvidencePage[] → StageResult<ProcedureScenario>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### releaseInstruction

Algorithm: Obtain qualified review of the exact variant, publish controlled copies, and coordinate acknowledgment or training. Retire outdated copies without erasing historical execution records and re-evaluate affected instructions when equipment or master procedures change.

Gate: Approved version and site applicability are visible at every point of use.

Contract: ``GoalContract + PolicySelection + Relevant<ProcedureScenario>[] + EvidencePage[] → StageResult<InstructionRelease>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
