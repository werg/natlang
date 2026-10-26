---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EngineeringImpact[] }
returns: StageResultChangeWorkPackage
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Build coordinated implementation packages covering procurement, stock, production, documentation, and field actions as applicable. Sequence changes to avoid incompatible mixed configurations and identify irreversible dispositions requiring separate authorization. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultChangeWorkPackage with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Feasible version transition and explicit handling of existing stock and in-flight work.
