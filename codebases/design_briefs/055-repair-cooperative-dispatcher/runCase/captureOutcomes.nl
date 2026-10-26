---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RepairWorkPlan[] }
returns: StageResultRepairOutcome
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare post-work checks and customer observations with the original symptoms. Record partial success, recurrence, and specialist referrals, then extract scoped lessons that improve future routing without treating one repair as universal proof of a cause. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRepairOutcome with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Outcome is verified and lessons retain applicability limits.
