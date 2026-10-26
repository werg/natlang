---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentationPatch[] }
returns: StageResultDocumentationRelease
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Re-extract changed promises, rerun examples and targeted behavior probes, and inspect linked pages for stale repetitions. Publish the reviewed bundle only when its examples and claims correspond to the same release version. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDocumentationRelease with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unresolved blocking mismatch or broken executable example in the released bundle.
