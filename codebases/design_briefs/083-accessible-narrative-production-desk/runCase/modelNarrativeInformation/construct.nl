---
description: Construct a NarrativeInformation draft for modelNarrativeInformation.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AccessibleProduction[] }
returns: DraftNarrativeInformation
---
Assemble the NarrativeInformation as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No information is invented beyond supplied media evidence or revealed prematurely.
