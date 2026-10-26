---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FabricationRequirement[] }
returns: StageResultFabricationConcept
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Propose distinct concepts using the shop’s supplied capabilities and approved materials. Explain appearance, cost, durability, and lead-time tradeoffs without claiming unsupported performance; preserve essential customer constraints across creative alternatives. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFabricationConcept with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Concept is feasible under documented shop capability and approved design constraints.
