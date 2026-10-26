---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultReuseSiteFact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile inventories, surveys, historical plans, and stakeholder descriptions into verified facts and unknowns. Distinguish current observations from outdated drawings and preserve uncertainties that could materially change feasibility or cost. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseSiteFact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No assumed structural or regulatory fact without supplied evidence.
