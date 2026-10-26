---
description: Construct a IssueReadiness draft for qualifyIssue.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EditorialTask[] }
returns: DraftIssueReadiness
---
Assemble the IssueReadiness as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Complete issue meets scope, rights, factual, and production gates.
