---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: GardenAgreement[] }
returns: StageResultGardenTask
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Turn accepted plans into bounded seasonal tasks, shared-tool reservations, and reminders. Adapt to changed availability or observations, preserve agreements across seasons, and record actual completion without treating missed tasks as automatic grounds for exclusion. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultGardenTask with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Resource scheduling and member consent remain current through replanning.
