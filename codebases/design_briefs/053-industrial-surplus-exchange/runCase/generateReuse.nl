---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseNeed[] }
returns: StageResultReuseMatch
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Propose conventional and inventive matches based on function rather than product names. Explain the adaptation mechanism and uncertainty, include multi-item bundles when useful, and reject creativity that depends on relaxing essential requirements or unapproved handling. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseMatch with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Essential fit is evidenced; speculative compatibility creates a qualification task, not an allocation.
