---
description: Construct a IssueCoverage draft for mapCoverage.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EditorialPitch[] }
returns: DraftIssueCoverage
---
Assemble the IssueCoverage as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Coverage choices follow the issue brief and available evidence.
