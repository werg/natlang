---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestReadiness[] }
returns: StageResultHarvestCommitment
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile delivery agreements and customer preferences to identify quantity, quality, and timing requirements. Separate permitted substitutions from assumptions and retain dependencies such as processing or storage windows that constrain otherwise feasible harvest choices. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHarvestCommitment with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every critical quality and substitution condition is explicit.
