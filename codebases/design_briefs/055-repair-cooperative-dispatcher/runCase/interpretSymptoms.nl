---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultRepairSymptom
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read the request and prior attempts to separate observations from the requester’s diagnosis. Identify approved handling restrictions and information needed for safe routing, without inventing technical causes or giving unqualified hazardous repair instructions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRepairSymptom with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Observations and inferred causes remain distinct; risk flags trigger specialist routing.
