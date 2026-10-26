---
description: Construct a MultilingualConcept draft for alignConcepts.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TermOccurrence[] }
returns: DraftMultilingualConcept
---
Assemble the MultilingualConcept as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Concept alignment is supported by definitions and counterexamples, not dictionary matching alone.
