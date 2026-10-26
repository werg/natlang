---
description: Construct a PreparationRequirement draft for recoverInstructions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AppointmentPreparation[] }
returns: DraftPreparationRequirement
---
Assemble the PreparationRequirement as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every consequential instruction and exception remains traceable to the approved source.
