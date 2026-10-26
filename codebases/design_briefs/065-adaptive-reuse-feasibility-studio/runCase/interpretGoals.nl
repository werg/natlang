---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseSiteFact[] }
returns: StageResultReuseGoal
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Translate stakeholder objectives into required functions and explicit tradeoffs. Separate mandatory supplied constraints from preferences, identify incompatible goals, and avoid choosing political or commercial priorities not authorized by the project brief. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseGoal with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Goals and constraints retain accountable owners and authority basis.
