---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: SourceFieldMeaning[] }
returns: StageResultMigrationMapping
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct conditional field and entity mappings that preserve domain meaning. Identify lossy transformations, required target extensions, and records needing manual resolution; do not invent missing facts to satisfy a stricter target schema. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMigrationMapping with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every mapping declares its domain, missing-value behavior, and information loss.
