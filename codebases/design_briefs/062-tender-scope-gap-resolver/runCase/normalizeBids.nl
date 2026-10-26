---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultBidScope
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read each bid, qualification, schedule, and addendum as a package. Extract functional scope rather than relying on trade headings, distinguish allowances from firm inclusions, and retain conditional exclusions that affect interfaces with other bidders. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultBidScope with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every scope statement retains its original qualification and bid version.
