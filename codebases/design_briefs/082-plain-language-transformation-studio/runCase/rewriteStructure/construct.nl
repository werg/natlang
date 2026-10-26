---
description: Construct a PlainLanguageDraft draft for rewriteStructure.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReaderTask[] }
returns: DraftPlainLanguageDraft
---
Assemble the PlainLanguageDraft as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Rewrite preserves every material condition, exception, and uncertainty.
