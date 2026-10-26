---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverRequirement[] }
returns: StageResultAssetIdentity
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile asset tags, drawing labels, manuals, and contractor naming conventions. Distinguish identical models at different locations, preserve serial and revision data, and flag ambiguous identity before attaching a certificate or manual to the wrong asset. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAssetIdentity with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Evidence attaches to the correct asset instance and revision.
