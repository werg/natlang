---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationMapping[] }
returns: StageResultMigrationDecision
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Apply mappings to bounded record groups, resolving semantic exceptions through evidence-backed decisions. Preserve identity and relationship provenance, isolate incompatible records, and avoid forcing all source variants into one target category for completion metrics. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMigrationDecision with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Accepted target records satisfy schema and source-meaning checks.
