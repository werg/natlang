# Decomposition: Habitat restoration planner

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Build staged, expert-reviewable restoration and monitoring plans that adapt to new observations without overstating ecological certainty.

Invariant: An unusual observation is not necessarily an error, an intervention’s ecological effect is uncertain, and activity completion does not prove environmental impact.

### interpretBaseline

Algorithm: Reconcile surveys, field notes, and historical records into a baseline with explicit spatial, temporal, and methodological limits. Distinguish observed absence from unsurveyed areas and preserve rare or uncertain observations for specialist review.

Gate: Baseline claims are limited to actual survey coverage and methods.

Contract: ``GoalContract + PolicySelection + Relevant<RestorationProject>[] + EvidencePage[] → StageResult<HabitatObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### translateObjectives

Algorithm: Convert landholder and ecological goals into observable objectives under supplied guidance. Separate desired long-term outcomes from near-term actions, identify conflicting uses, and preserve uncertainty about causal pathways rather than promising guaranteed ecological results.

Gate: Objectives have measurable indicators and accountable expert interpretation.

Contract: ``GoalContract + PolicySelection + Relevant<HabitatObservation>[] + EvidencePage[] → StageResult<RestorationObjective>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructInterventions

Algorithm: Propose staged interventions from approved guidance and known site conditions. Explain the hypothesized mechanism, dependencies, resource needs, and stop conditions; route unfamiliar or potentially harmful interventions to qualified specialists instead of improvising.

Gate: Known ecological and access constraints respected; no unsupported intervention prescription.

Contract: ``GoalContract + PolicySelection + Relevant<RestorationObjective>[] + EvidencePage[] → StageResult<RestorationOption>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planMonitoring

Algorithm: Design monitoring that can distinguish progress, nonresponse, and unintended effects using supplied methods. Specify comparable locations, timing, and effort, and define what evidence would justify continuing, revising, or stopping an intervention.

Gate: Monitoring can inform a real decision and does not confuse changing effort with changing conditions.

Contract: ``GoalContract + PolicySelection + Relevant<RestorationOption>[] + EvidencePage[] → StageResult<MonitoringTask>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### adaptManagement

Algorithm: Interpret new observations against baseline and monitoring rules, including alternative explanations such as seasonality or survey changes. Propose bounded plan revisions, preserve unaffected work, and obtain expert approval before changing interventions.

Gate: Adaptive changes retain evidence, uncertainty, and current access or intervention authority.

Contract: ``GoalContract + PolicySelection + Relevant<MonitoringTask>[] + EvidencePage[] → StageResult<RestorationRevision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
