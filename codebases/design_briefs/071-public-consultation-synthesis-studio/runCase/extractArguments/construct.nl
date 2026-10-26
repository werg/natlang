---
description: Construct a ConsultationArgument draft for extractArguments.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: Consultation[] }
returns: DraftConsultationArgument
---
Assemble the ConsultationArgument as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Each argument is attributable to its submission without invented motives.
