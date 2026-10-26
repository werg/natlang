---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDocumentedPromise
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Extract checkable statements about inputs, outputs, defaults, errors, and supported workflows from documentation and examples. Preserve version and audience scope, distinguishing normative promises from illustrative suggestions or historical notes. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDocumentedPromise with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every promise has an applicable version and source span.
