---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultSiteInstruction
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile site notes, messages, and drawing annotations to identify the actual requested action and its issuer. Distinguish clarification, proposed change, and authorized instruction; preserve conflicting versions and do not infer commercial authority from site presence. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultSiteInstruction with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Instruction scope and authority are separately evidenced.
