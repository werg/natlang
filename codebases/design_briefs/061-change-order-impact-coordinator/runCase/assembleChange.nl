---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeOption[] }
returns: StageResultChangePackage
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Prepare a coordinated scope, cost, timing, and responsibility package with the relevant evidence. Separate agreed changes from disputed claims, bind approvals to the exact revision, and update downstream plans only after the appropriate authorities accept the change. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultChangePackage with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unapproved scope or schedule change silently enters the committed baseline.
