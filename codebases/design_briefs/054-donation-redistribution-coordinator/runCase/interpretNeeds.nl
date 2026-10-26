---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DonationOffer[] }
returns: StageResultRecipientNeed
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Recover recipient needs from descriptions and prior confirmations without inferring sensitive circumstances. Distinguish necessary features from preferences, record accessibility and timing constraints explicitly supplied by recipients, and preserve consent for contact or delivery. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRecipientNeed with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Suitability criteria are recipient-grounded and not based on stereotypes.
