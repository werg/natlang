---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestCommitment[] }
returns: StageResultHarvestMatch
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare lot characteristics and readiness with customer requirements, identifying compatible combinations and gaps. Preserve uncertain quality findings and request targeted verification rather than allocating a lot solely because its crop name matches. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHarvestMatch with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Allocated lots satisfy confirmed essential requirements.
