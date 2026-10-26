---
description: Construct a TrainingMechanism draft for recoverLearningMechanism.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TrainingAdaptation[] }
returns: DraftTrainingMechanism
---
Assemble the TrainingMechanism as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every scenario has an explicit instructional mechanism before adaptation.
