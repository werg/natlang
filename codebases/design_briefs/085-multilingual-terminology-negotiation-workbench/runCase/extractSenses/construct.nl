---
description: Construct a TermOccurrence draft for extractSenses.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TerminologyProject[] }
returns: DraftTermOccurrence
---
Assemble the TermOccurrence as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every sense has contextual evidence.
