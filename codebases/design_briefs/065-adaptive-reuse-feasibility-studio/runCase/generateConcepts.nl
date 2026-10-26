---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseGoal[] }
returns: StageResultReuseConcept
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Invent distinct reuse concepts that exploit the building’s known characteristics while respecting supplied constraints. Describe functional organization and phasing, preserve uncertainty about unverified conditions, and avoid presenting conceptual layouts as certified engineering designs. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseConcept with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Concepts satisfy known hard constraints and expose every consequential assumption.
