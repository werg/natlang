---
description: Construct a EventCommitment draft for recoverEventPromises.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LiveEvent[] }
returns: DraftEventCommitment
---
Assemble the EventCommitment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Commitment priority follows supplied authority and actual promises.
