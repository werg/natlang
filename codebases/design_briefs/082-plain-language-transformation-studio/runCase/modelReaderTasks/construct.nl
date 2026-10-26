---
description: Construct a ReaderTask draft for modelReaderTasks.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MeaningUnit[] }
returns: DraftReaderTask
---
Assemble the ReaderTask as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Reader tasks cover all consequential uses of the document.
