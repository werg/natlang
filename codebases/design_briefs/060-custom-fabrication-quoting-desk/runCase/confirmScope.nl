---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FabricationEstimate[] }
returns: StageResultFabricationAgreement
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Present drawings or structured specifications, finish samples, assumptions, and acceptance criteria for customer confirmation. Bind the quote to the accepted revision, then create production work only after material scope and commercial terms are approved. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFabricationAgreement with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Quoted object, accepted specification, and production route are mutually consistent.
