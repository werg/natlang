---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentationMismatch[] }
returns: StageResultDocumentationPatch
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Draft coordinated changes to pages, examples, tests, and release notes according to the reviewed repair direction. Preserve unaffected claims, update cross-links and terminology, and keep code examples executable under their stated environment. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDocumentationPatch with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact patch bases and semantic preservation checks pass.
