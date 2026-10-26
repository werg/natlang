---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CanonicalEntity[] }
returns: StageResultIdentityRevision
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Process new evidence and reviewer corrections as reversible identity changes. Trace downstream uses, prepare split or merge migrations, and preserve old identifiers as redirects with explicit validity rather than silently changing historical joins. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIdentityRevision with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Consumer impact and reversibility verified before promoting an identity revision.
