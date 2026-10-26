---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultGardenPreference
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read member requests and prior agreements to identify desired uses, availability, and explicitly stated access needs. Separate preferences from accepted communal obligations and avoid inferring personal circumstances or assigning work without consent. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultGardenPreference with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Member commitments are explicit and attributable.
