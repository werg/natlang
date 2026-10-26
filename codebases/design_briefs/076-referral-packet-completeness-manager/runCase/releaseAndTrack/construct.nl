---
description: Construct a ReferralPacket draft for releaseAndTrack.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReferralRequest[] }
returns: DraftReferralPacket
---
Assemble the ReferralPacket as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No claim of appointment or clinical acceptance from a transport receipt alone.
