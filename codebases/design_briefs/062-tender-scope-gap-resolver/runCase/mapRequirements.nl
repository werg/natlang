---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BidScope[] }
returns: StageResultScopeCoverage
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare required functions with bid inclusions and exclusions, recognizing different descriptions of the same work. Separate supply, installation, commissioning, and certification responsibilities instead of assuming one broad line covers all phases. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultScopeCoverage with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Coverage is phase-specific and evidence-backed.
