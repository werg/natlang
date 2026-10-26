---
description: Construct a SubmissionRelation draft for linkCampaigns.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ConsultationArgument[] }
returns: DraftSubmissionRelation
---
Assemble the SubmissionRelation as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Template use does not invalidate a participant’s submission or erase unique content.
