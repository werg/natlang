---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PlotConstraint[] }
returns: StageResultGardenPlan
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Propose varied planting and shared-space concepts consistent with supplied guidance and member goals. Include creative communal uses, preserve access paths and resource limits, and identify tradeoffs rather than optimizing only one member’s preferences. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultGardenPlan with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Known hard constraints and approved garden rules remain satisfied.
