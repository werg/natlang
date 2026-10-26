---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeApplicability[] }
returns: StageResultEngineeringImpact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Traverse bills of material, interfaces, process instructions, and supplier commitments through exact dependency graphs. Judge whether each indirect dependency is materially affected and retain uncertain impacts for engineering review instead of broad automatic replacement. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEngineeringImpact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No missed critical dependency or unsupported global invalidation.
