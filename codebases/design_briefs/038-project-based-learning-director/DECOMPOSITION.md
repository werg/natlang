# Decomposition: Project-based learning director

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Build and adapt a coherent learning project with roles, milestones, creative complications, and evidence of intended competencies.

Invariant: A wrong answer may reflect ambiguity rather than misconception, fluent explanation is not demonstrated competence, and accommodations must not be inferred from sensitive attributes.

### translateGoals

Algorithm: Interpret broad educational goals as observable project evidence and identify prerequisite skills. Separate the project’s fictional or practical theme from the competencies it must exercise, and preserve constraints on resources and learner workload.

Gate: Every goal has a feasible demonstration within the project scope.

Contract: ``GoalContract + PolicySelection + Relevant<LearningProject>[] + EvidencePage[] → StageResult<ProjectLearningGoal>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### designProject

Algorithm: Invent a coherent setting and challenge that naturally requires the target skills. Design meaningful roles and alternative solution paths, avoid making one learner a passive spectator, and include authentic constraints without introducing inaccessible prerequisites.

Gate: All roles contribute substantive evidence and the challenge is solvable with available resources.

Contract: ``GoalContract + PolicySelection + Relevant<ProjectLearningGoal>[] + EvidencePage[] → StageResult<ProjectConcept>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### sequenceMilestones

Algorithm: Decompose the project into milestones with dependencies, feedback points, and explicit evidence criteria. Allocate time and resources through exact services, identify critical bottlenecks, and preserve room for learner choice within the required structure.

Gate: Acyclic dependency graph, feasible workload, and complete competency coverage.

Contract: ``GoalContract + PolicySelection + Relevant<ProjectConcept>[] + EvidencePage[] → StageResult<ProjectMilestone>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### adaptScaffolding

Algorithm: Interpret progress artifacts to identify where support is needed without doing the substantive work for learners. Provide targeted examples, prompts, or task decomposition, and remove support when evidence shows independent capability.

Gate: Adaptation preserves assessment validity and does not substitute generated work for learner evidence.

Contract: ``GoalContract + PolicySelection + Relevant<ProjectMilestone>[] + EvidencePage[] → StageResult<ScaffoldingPlan>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### introduceComplications

Algorithm: Introduce a creative but fair complication only after checking remaining time, resources, and prerequisites. Offer multiple recovery paths, revise logistics, and preserve the intended competencies rather than making the project impossible or changing its purpose.

Gate: Revised project remains feasible and all mandatory learning goals survive.

Contract: ``GoalContract + PolicySelection + Relevant<ScaffoldingPlan>[] + EvidencePage[] → StageResult<ProjectRevision>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
