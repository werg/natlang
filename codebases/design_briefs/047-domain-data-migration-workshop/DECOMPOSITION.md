# Decomposition: Domain data migration workshop

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Build reversible, evidence-backed semantic mappings from messy source systems to a target domain model.

Invariant: A matching label is not semantic equivalence, a missing value may carry meaning, and exact aggregation is correct only after the intended population and predicates are established.

### recoverSourceMeaning

Algorithm: Reconcile dictionaries, sample records, application notes, and domain-owner explanations. Identify fields whose meaning depends on record type or history, and distinguish null, not applicable, unknown, and deliberately withheld values before proposing transformations.

Gate: No target mapping based solely on field names or observed value formats.

Contract: ``GoalContract + PolicySelection + Relevant<MigrationProject>[] + EvidencePage[] → StageResult<SourceFieldMeaning>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### designMappings

Algorithm: Construct conditional field and entity mappings that preserve domain meaning. Identify lossy transformations, required target extensions, and records needing manual resolution; do not invent missing facts to satisfy a stricter target schema.

Gate: Every mapping declares its domain, missing-value behavior, and information loss.

Contract: ``GoalContract + PolicySelection + Relevant<SourceFieldMeaning>[] + EvidencePage[] → StageResult<MigrationMapping>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### resolveRecords

Algorithm: Apply mappings to bounded record groups, resolving semantic exceptions through evidence-backed decisions. Preserve identity and relationship provenance, isolate incompatible records, and avoid forcing all source variants into one target category for completion metrics.

Gate: Accepted target records satisfy schema and source-meaning checks.

Contract: ``GoalContract + PolicySelection + Relevant<MigrationMapping>[] + EvidencePage[] → StageResult<MigrationDecision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### rehearseMigration

Algorithm: Materialize a staged target snapshot, validate referential integrity and business invariants, and compare representative source-target round trips. Rehearse rollback and delta capture so cutover does not lose changes made after the initial extraction.

Gate: Exact counts, relationships, conservation constraints, and reversible mappings verified.

Contract: ``GoalContract + PolicySelection + Relevant<MigrationDecision>[] + EvidencePage[] → StageResult<MigrationRehearsal>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateCutover

Algorithm: Freeze the approved mapping and cutover scope, apply final source deltas, and reconcile target results before switching consumers. Track quarantined records and post-cutover corrections explicitly, retaining a safe rollback or forward-repair plan for irreversible changes.

Gate: No unaccounted source record or unapproved mapping change during cutover.

Contract: ``GoalContract + PolicySelection + Relevant<MigrationRehearsal>[] + EvidencePage[] → StageResult<MigrationCutover>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
