---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CategoryFit[] }
returns: StageResultTaxonomyDefect
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Analyze repeated fit problems to identify structural defects. Separate insufficient annotation guidance from real conceptual change, inspect negative examples, and avoid proliferating categories for every rare wording variant. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTaxonomyDefect with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Proposed structural changes are supported by multiple informative examples or a decisive new requirement.
