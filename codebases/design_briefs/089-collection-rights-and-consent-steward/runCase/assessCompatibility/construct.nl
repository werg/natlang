---
description: Construct a RightsFinding draft for assessCompatibility.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ProposedCollectionUse[] }
returns: DraftRightsFinding
---
Assemble the RightsFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every allowed use has positive authority and all applicable conditions are satisfied.
