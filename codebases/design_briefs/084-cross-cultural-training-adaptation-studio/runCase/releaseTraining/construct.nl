---
description: Construct a AdaptedTrainingRelease draft for releaseTraining.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AdaptationTest[] }
returns: DraftAdaptedTrainingRelease
---
Assemble the AdaptedTrainingRelease as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Approved audience scope and preserved learning objectives are explicit.
