---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RecipientNeed[] }
returns: StageResultDonationBundle
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Assemble useful combinations rather than allocating isolated items that cannot serve the recipient’s purpose. Propose acceptable substitutions, identify missing components, and ensure the bundle’s condition and practical requirements match the confirmed need. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDonationBundle with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Bundle is usable under the recipient’s stated constraints.
