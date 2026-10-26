---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EvidenceTask[] }
returns: StageResultPortfolioDisposition
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Assemble a reviewable recommendation with evidence, gaps, and proposed next work. Allow the learner to correct factual attribution, preserve assessor disagreements, and record the authorized decision separately from the program’s recommendation. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPortfolioDisposition with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: High-stakes outcomes require accountable assessor approval and an appeal path.
