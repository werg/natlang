---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScopeClarification[] }
returns: StageResultResponsibilityMatrix
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Assemble a reconciled responsibility matrix and commercial comparison using accepted clarifications. Recheck all required functions and dependencies, expose residual gaps, and bind the selection packet to the exact bid editions reviewed. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultResponsibilityMatrix with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No silent assignment of unaccepted obligations and no uncovered mandatory function.
