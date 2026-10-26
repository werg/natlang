---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultBehavioralRequirement
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret prose requirements, bug reports, and relevant discussion to identify observable behavior, preconditions, and exclusions. Separate implementation suggestions from requirements and preserve ambiguous expectations for owner resolution before encoding them as tests. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultBehavioralRequirement with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every assertion traces to a requirement or explicitly approved interpretation.
