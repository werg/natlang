---
description: Construct a VolunteerCapability draft for interpretProfiles.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MissionRequirement[] }
returns: DraftVolunteerCapability
---
Assemble the VolunteerCapability as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Capability and consent are explicit and current.
