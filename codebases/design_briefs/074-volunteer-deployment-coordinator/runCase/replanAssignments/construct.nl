---
description: Construct a VolunteerRevision draft for replanAssignments.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: VolunteerBriefing[] }
returns: DraftVolunteerRevision
---
Assemble the VolunteerRevision as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Replanned teams retain required coverage, supervision, and current consent.
