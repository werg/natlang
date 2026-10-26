---
description: Construct a ServicePathway draft for identifyPathways.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ServiceSituation[] }
returns: DraftServicePathway
---
Assemble the ServicePathway as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every pathway follows current supplied rules and clearly states unresolved conditions.
