---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultBehavioralChange
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read release diffs, requirements, and observed behavior to identify changed semantics, defaults, timing, and error handling. Distinguish internal refactoring from externally meaningful changes even when schemas and type signatures remain compatible. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultBehavioralChange with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every claimed behavioral change has evidence or a targeted verification task.
