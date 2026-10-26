---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ConstructionImpact[] }
returns: StageResultChangeOption
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct feasible alternatives such as resequencing, substitution, or phased completion within approved design constraints. Obtain exact estimates from authorized services, disclose assumptions, and preserve professional review requirements for structural or safety consequences. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultChangeOption with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Option is technically reviewable, commercially explicit, and dependency-feasible.
