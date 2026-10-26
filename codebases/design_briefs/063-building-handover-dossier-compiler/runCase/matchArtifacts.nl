---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssetIdentity[] }
returns: StageResultHandoverEvidence
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare submitted manuals, inspection records, certificates, and training logs with the actual requirement. Distinguish a blank form from completed evidence and a generic manual from the installed configuration; retain missing signatures or scope mismatches. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHandoverEvidence with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Artifact presence alone does not satisfy substantive evidence requirements.
