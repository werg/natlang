---
description: Construct a ReferralRequest draft for requestMaterial.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReferralGap[] }
returns: DraftReferralRequest
---
Assemble the ReferralRequest as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Requests are minimal, authorized, and tied to a specific requirement.
