---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultIncidentImpact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Combine alerts, user reports, and service telemetry to describe observed impact and uncertainty. Separate symptoms from causes and distinguish missing telemetry from healthy behavior; identify the service outcomes that recovery must actually restore. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIncidentImpact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Impact statements cite observations and preserve unknown coverage.
