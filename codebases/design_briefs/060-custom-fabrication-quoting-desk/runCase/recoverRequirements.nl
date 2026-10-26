---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultFabricationRequirement
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret sketches, descriptions, and intended use to distinguish functional requirements from aesthetic preferences and unstated assumptions. Identify consequential missing dimensions or loads and request clarification rather than inventing specifications that determine manufacturability or safety. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFabricationRequirement with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Critical requirements are confirmed or explicitly block a firm quote.
