---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProjectConcept[] }
returns: StageResultProjectMilestone
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Decompose the project into milestones with dependencies, feedback points, and explicit evidence criteria. Allocate time and resources through exact services, identify critical bottlenecks, and preserve room for learner choice within the required structure. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProjectMilestone with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Acyclic dependency graph, feasible workload, and complete competency coverage.
