---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingCommitment[] }
returns: StageResultShippingAlternative
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Generate permitted rerouting, split-shipment, consolidation, or substitution options. Explain which commitments each option preserves or changes, and exclude alternatives violating handling requirements or relying on unverified carrier capacity. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultShippingAlternative with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Essential constraints remain intact; changed promises require explicit customer or owner approval.
