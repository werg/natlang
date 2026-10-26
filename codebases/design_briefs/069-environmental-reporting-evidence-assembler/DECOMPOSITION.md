# Decomposition: Environmental reporting evidence assembler

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Connect supplied reporting obligations to measurements and operational evidence without double-counting or unsupported impact claims.

Invariant: An unusual observation is not necessarily an error, an intervention’s ecological effect is uncertain, and activity completion does not prove environmental impact.

### interpretBoundaries

Algorithm: Read the supplied reporting framework, organizational boundary, and amendments to determine included activities and required measures. Distinguish ownership, operational control, and other supplied boundary concepts rather than choosing whichever yields a favorable result.

Gate: Boundary interpretation is explicit and approved before aggregation.

Contract: ``GoalContract + PolicySelection + Relevant<EnvironmentalReport>[] + EvidencePage[] → StageResult<ReportingBoundary>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### mapMeasurements

Algorithm: Reconcile measurements, invoices, contractor statements, and field records with their actual methods and periods. Preserve estimates versus direct measurements, identify missing coverage, and retain original units and uncertainty for exact conversion services.

Gate: Every value has method, period, unit, and provenance.

Contract: ``GoalContract + PolicySelection + Relevant<ReportingBoundary>[] + EvidencePage[] → StageResult<EnvironmentalMeasure>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### reconcileAttribution

Algorithm: Allocate shared activities using supplied rules and exact calculations, checking for duplicate records or overlapping contractor reports. Distinguish a transferred responsibility from a duplicated physical activity and preserve unresolved attribution rather than balancing by assumption.

Gate: Exact totals and no double-counting across reporting boundaries.

Contract: ``GoalContract + PolicySelection + Relevant<EnvironmentalMeasure>[] + EvidencePage[] → StageResult<EnvironmentalAllocation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### substantiateClaims

Algorithm: Compare proposed narrative claims with actual measured outcomes and activity evidence. Separate completion of an initiative from demonstrated environmental benefit, qualify estimated results, and reject persuasive claims unsupported by the reporting data.

Gate: Each quantitative and impact claim is supported at the stated scope.

Contract: ``GoalContract + PolicySelection + Relevant<EnvironmentalAllocation>[] + EvidencePage[] → StageResult<EnvironmentalClaim>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### assembleSubmission

Algorithm: Build the required schedules, methodology notes, and narrative with consistent quantities and boundaries. Present gaps and restatements clearly, obtain responsible review, and preserve the exact submitted edition and subsequent corrections.

Gate: Narrative, calculations, methods, and approved boundary all agree.

Contract: ``GoalContract + PolicySelection + Relevant<EnvironmentalClaim>[] + EvidencePage[] → StageResult<EnvironmentalSubmission>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
