---
description: Construct a ConsultationReport draft for verifySynthesis.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ConsultationResponseOption[] }
returns: DraftConsultationReport
---
Assemble the ConsultationReport as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No material theme omission, misleading denominator, or privacy leak in published evidence.
