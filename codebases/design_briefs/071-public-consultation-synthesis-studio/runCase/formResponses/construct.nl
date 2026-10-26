---
description: Construct a ConsultationResponseOption draft for formResponses.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ConsultationTheme[] }
returns: DraftConsultationResponseOption
---
Assemble the ConsultationResponseOption as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Response accurately represents the concern and does not claim unsupported public consensus.
