---
description: Construct a EscapeConstraint draft for modelExperience.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EscapeRoomProject[] }
returns: DraftEscapeConstraint
---
Assemble the EscapeConstraint as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No real confinement, unsafe mechanism, or inaccessible mandatory path under supplied requirements.
