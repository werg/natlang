---
description: Construct a TerminologyDecision draft for negotiateDecision.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TermOption[] }
returns: DraftTerminologyDecision
---
Assemble the TerminologyDecision as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Decision authority and applicability are explicit; unresolved conceptual conflict remains visible.
