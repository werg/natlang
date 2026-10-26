---
description: Construct a LocaleQualityFinding draft for validateLocale.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LocalizedSegment[] }
returns: DraftLocaleQualityFinding
---
Assemble the LocaleQualityFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact structural gates pass and no unresolved material semantic defect remains.
