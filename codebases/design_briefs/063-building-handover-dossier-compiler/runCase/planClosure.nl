---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverEvidence[] }
returns: StageResultHandoverClosureTask
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Group gaps into actionable requests for contractors, inspectors, or facilities staff. Prioritize dependencies affecting safe or practical use under supplied rules, and preserve accepted temporary arrangements with owners and expiry rather than hiding them as complete. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHandoverClosureTask with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every mandatory gap has an accountable closure path or explicit escalation.
