---
description: Construct a JourneyLink draft for linkJourneyEvents.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ExperienceEpisode[] }
returns: DraftJourneyLink
---
Assemble the JourneyLink as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Patient isolation and evidence-backed episode linkage.
