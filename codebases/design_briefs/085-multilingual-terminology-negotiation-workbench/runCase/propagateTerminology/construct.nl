---
description: Construct a TerminologyImpact draft for propagateTerminology.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TerminologyDecision[] }
returns: DraftTerminologyImpact
---
Assemble the TerminologyImpact as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No global search-and-replace across different senses; affected content passes fidelity checks.
