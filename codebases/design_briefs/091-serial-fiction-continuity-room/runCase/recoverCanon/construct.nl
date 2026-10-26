---
description: Construct a CanonFact draft for recoverCanon.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: StorySeries[] }
returns: DraftCanonFact
---
Assemble the CanonFact as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every canon fact has source and status, including belief versus world fact.
