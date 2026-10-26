---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDisruptionImpact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read carrier notices and operational reports to determine what is unavailable, for how long, and with what certainty. Distinguish a delayed service from a closed route and preserve exceptions that may allow some shipment classes to continue. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDisruptionImpact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Scope is evidence-backed and uncertain reopening times are not treated as confirmed capacity.
