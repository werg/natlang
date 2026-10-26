---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityCandidate[] }
returns: StageResultIdentityJudgment
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare names, attributes, chronology, and context while distinguishing identity from affiliation or succession. Treat missing attributes as unknown rather than matching, and record both positive support and hard cannot-link evidence. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIdentityJudgment with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Same-entity judgments require compatible identity evidence, not shared names alone.
