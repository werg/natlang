---
description: Construct a MediationPosition draft for mapPositions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ExistingAgreement[] }
returns: DraftMediationPosition
---
Assemble the MediationPosition as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Each position is confirmed or clearly marked as an unconfirmed interpretation.
