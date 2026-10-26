---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScaffoldingPlan[] }
returns: StageResultProjectRevision
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Introduce a creative but fair complication only after checking remaining time, resources, and prerequisites. Offer multiple recovery paths, revise logistics, and preserve the intended competencies rather than making the project impossible or changing its purpose. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProjectRevision with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Revised project remains feasible and all mandatory learning goals survive.
