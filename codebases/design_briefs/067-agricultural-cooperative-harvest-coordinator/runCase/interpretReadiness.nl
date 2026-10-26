---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultHarvestReadiness
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read grower notes and approved assessment records to identify likely readiness, quality, and uncertainty. Distinguish tentative expectations from confirmed availability and preserve variety or lot differences that affect harvest timing and customer suitability. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHarvestReadiness with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Readiness is evidence-backed and does not become a guarantee from informal wording.
