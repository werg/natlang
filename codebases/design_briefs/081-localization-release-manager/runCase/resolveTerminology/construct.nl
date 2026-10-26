---
description: Construct a LocaleTermDecision draft for resolveTerminology.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LocalizationSegment[] }
returns: DraftLocaleTermDecision
---
Assemble the LocaleTermDecision as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Term choice is concept- and context-specific, with versioned authority.
