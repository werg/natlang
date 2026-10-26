---
description: Construct a ExperienceOutcome draft for reviewResults.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ServiceExperiment[] }
returns: DraftExperienceOutcome
---
Assemble the ExperienceOutcome as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Reported improvements match exact measures and preserve limits and counterevidence.
