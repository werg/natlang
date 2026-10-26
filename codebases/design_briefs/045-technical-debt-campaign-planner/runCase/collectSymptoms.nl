---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDebtSymptom
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Process developer reports, incidents, and issue histories to identify recurring friction. Separate specific observed costs from broad complaints, deduplicate repeated descriptions of one event, and preserve minority workflows that may face severe but infrequent problems. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDebtSymptom with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Symptom frequency and impact are grounded in observable evidence.
