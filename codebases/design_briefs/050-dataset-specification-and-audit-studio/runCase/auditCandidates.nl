---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnnotationRule[] }
returns: StageResultDatasetRecordAudit
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Inspect candidate records for task relevance, data quality, provenance, consent, and leakage risks. Distinguish missing coverage from invalid records, and quarantine material whose rights or source integrity cannot be established. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDatasetRecordAudit with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No release candidate without verified provenance and permitted intended use.
