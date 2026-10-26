---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RecoveryStep[] }
returns: StageResultRestorationFinding
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Recheck the affected user-visible functions and representative cohorts after recovery actions. Distinguish reduced alerts from actual restoration, retain residual degradation, and schedule observation long enough to detect recurrence under the supplied incident policy. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRestorationFinding with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Closure requires outcome verification, not merely green infrastructure metrics.
