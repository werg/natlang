---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: SurplusOffer[] }
returns: StageResultReuseNeed
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Recover the actual purpose behind recipient requests, separating essential performance from preferred form. Identify acceptable adaptations and nonnegotiable constraints, including evidence or specialist approval needed for unfamiliar materials or equipment. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseNeed with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Recipient confirms constraints that determine suitability or safety.
