---
description: Construct a TheatreConstraint draft for modelProduction.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TheatreScene[] }
returns: DraftTheatreConstraint
---
Assemble the TheatreConstraint as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Required resources and consent conditions are explicit and current.
