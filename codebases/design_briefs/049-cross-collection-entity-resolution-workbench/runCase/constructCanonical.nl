---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityCluster[] }
returns: StageResultCanonicalEntity
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Assemble a canonical view without overwriting source records or selecting one source globally for every field. Resolve attributes using field-specific authority and effective time, preserve aliases and conflicting claims, and separate historical from current values. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultCanonicalEntity with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Canonical values remain attributable and do not erase meaningful disagreement.
