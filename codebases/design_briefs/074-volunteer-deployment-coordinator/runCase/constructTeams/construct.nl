---
description: Construct a VolunteerTeam draft for constructTeams.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: VolunteerCapability[] }
returns: DraftVolunteerTeam
---
Assemble the VolunteerTeam as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: All required capabilities and consent conditions satisfied or the mission remains blocked.
