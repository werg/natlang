---
description: Construct a ExperienceEpisode draft for extractExperiences.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ExperienceStudy[] }
returns: DraftExperienceEpisode
---
Assemble the ExperienceEpisode as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Findings stay within observed administrative experience.
