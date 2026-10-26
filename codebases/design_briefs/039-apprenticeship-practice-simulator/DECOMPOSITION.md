# Decomposition: Apprenticeship practice simulator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Run realistic workplace practice with hidden scenario state, approved procedures, and evidence-based feedback.

Invariant: A wrong answer may reflect ambiguity rather than misconception, fluent explanation is not demonstrated competence, and accommodations must not be inferred from sensitive attributes.

### authorSituation

Algorithm: Construct a realistic situation requiring the supplied procedure and meaningful information gathering. Separate facts visible at the start from facts revealed by appropriate questions, and include complications that test judgment without requiring unauthorized actions.

Gate: Scenario has a feasible compliant path and no hidden prerequisite the learner cannot discover.

Contract: ``GoalContract + PolicySelection + Relevant<PracticeSession>[] + EvidencePage[] → StageResult<PracticeScenario>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretAction

Algorithm: Interpret the learner’s natural-language response against the available action set. Distinguish a proposed action from an executed one, ask for clarification when ambiguity affects consequences, and avoid rewarding jargon without the corresponding operational behavior.

Gate: Action maps to a permitted typed operation or remains unexecuted.

Contract: ``GoalContract + PolicySelection + Relevant<PracticeScenario>[] + EvidencePage[] → StageResult<LearnerAction>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### advanceWorld

Algorithm: Apply the exact simulator’s legal transition rules, then formulate context-appropriate responses from actors who know only their own information. Preserve consequences across turns and do not reveal hidden facts merely to help the learner succeed.

Gate: Resource, knowledge, and procedure invariants remain valid after every transition.

Contract: ``GoalContract + PolicySelection + Relevant<LearnerAction>[] + EvidencePage[] → StageResult<PracticeTransition>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### introduceChallenge

Algorithm: Select a complication appropriate to the learner’s progress and learning objective. Ensure it changes a meaningful decision, preserves a safe recovery path, and does not retroactively invalidate an action that was correct under earlier visible facts.

Gate: Complication is fair, procedurally relevant, and reproducible from the scenario seed.

Contract: ``GoalContract + PolicySelection + Relevant<PracticeTransition>[] + EvidencePage[] → StageResult<PracticeComplication>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### debriefPerformance

Algorithm: Compare the observable trajectory with procedure requirements and scenario facts. Explain specific effective and ineffective decisions, distinguish outcome luck from sound process, and propose a next scenario targeting demonstrated gaps rather than generic advice.

Gate: Feedback cites actual actions and does not infer unobserved intentions.

Contract: ``GoalContract + PolicySelection + Relevant<PracticeComplication>[] + EvidencePage[] → StageResult<PracticeFeedback>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
