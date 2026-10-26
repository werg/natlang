---
description: Construct a MissionRequirement draft for interpretMission.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: VolunteerMission[] }
returns: DraftMissionRequirement
---
Assemble the MissionRequirement as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Requirements are task-based and not inferred from volunteer demographics.
