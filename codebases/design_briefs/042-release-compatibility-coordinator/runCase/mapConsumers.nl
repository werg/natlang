---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehavioralChange[] }
returns: StageResultConsumerDependency
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Relate integrations, customer configurations, examples, and runbooks to the changed behavior. Identify implicit reliance such as ordering or retry assumptions, and preserve uncertain dependencies for targeted replay rather than declaring them unaffected. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultConsumerDependency with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Dependency claims are specific to behavior, not mere package-name overlap.
