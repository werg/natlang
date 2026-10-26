---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: GardenPreference[] }
returns: StageResultPlotConstraint
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile plot records, field observations, and supplied gardening guidance to describe known conditions and constraints. Preserve uncertainty about soil or other untested properties, and route consequential interventions to appropriate local expertise. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPlotConstraint with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Planning uses verified or explicitly uncertain conditions, not invented agronomic facts.
