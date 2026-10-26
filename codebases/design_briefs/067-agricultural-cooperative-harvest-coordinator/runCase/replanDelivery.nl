---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestPlan[] }
returns: StageResultHarvestRevision
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Process changed field observations, resource failures, or customer responses as scoped deltas. Reallocate only affected lots, obtain consent for material substitutions, and reconcile actual harvested and delivered quantities before declaring commitments fulfilled. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHarvestRevision with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Revised commitments are accepted and actual quantities remain traceable by lot.
