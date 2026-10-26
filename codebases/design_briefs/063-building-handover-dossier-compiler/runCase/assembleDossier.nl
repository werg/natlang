---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverClosureTask[] }
returns: StageResultHandoverDossier
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Produce a navigable asset-linked dossier and a separate outstanding-obligations register. Verify search, links, file integrity, and recipient access, then record documentary acceptance independently from any qualified technical approval or occupancy decision. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHandoverDossier with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No claim of technical readiness based solely on a complete document bundle.
