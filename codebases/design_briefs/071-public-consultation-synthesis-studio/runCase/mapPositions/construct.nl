---
description: Construct a ConsultationTheme draft for mapPositions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: SubmissionRelation[] }
returns: DraftConsultationTheme
---
Assemble the ConsultationTheme as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every substantive argument has a theme or explicit out-of-scope disposition.
