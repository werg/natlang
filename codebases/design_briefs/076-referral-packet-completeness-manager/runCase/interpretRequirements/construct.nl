---
description: Construct a ReferralRequirement draft for interpretRequirements.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReferralCase[] }
returns: DraftReferralRequirement
---
Assemble the ReferralRequirement as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Requirements are current, service-specific, and administrative in scope.
