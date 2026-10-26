---
description: Construct a PreparationStatus draft for trackAcknowledgment.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PreparationQuestion[] }
returns: DraftPreparationStatus
---
Assemble the PreparationStatus as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No false readiness claim and no unapproved clinical recommendation.
