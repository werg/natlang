---
description: Construct a PlayerIntent draft for interpretIntent.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: GameWorld[] }
returns: DraftPlayerIntent
---
Assemble the PlayerIntent as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Intent resolves to a legal action proposal or remains unexecuted.
