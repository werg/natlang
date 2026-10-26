---
description: Construct a ReferralGap draft for assessCompleteness.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReferralRecordMatch[] }
returns: DraftReferralGap
---
Assemble the ReferralGap as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every administrative requirement is met or explicitly unresolved.
