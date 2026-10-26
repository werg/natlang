---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EnvironmentalClaim[] }
returns: StageResultEnvironmentalSubmission
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Build the required schedules, methodology notes, and narrative with consistent quantities and boundaries. Present gaps and restatements clearly, obtain responsible review, and preserve the exact submitted edition and subsequent corrections. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEnvironmentalSubmission with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Narrative, calculations, methods, and approved boundary all agree.
