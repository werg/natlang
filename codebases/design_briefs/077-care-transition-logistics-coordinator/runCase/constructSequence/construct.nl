---
description: Construct a TransitionTask draft for constructSequence.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TransitionParticipant[] }
returns: DraftTransitionTask
---
Assemble the TransitionTask as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: All logistical prerequisites are feasible and traceable to the approved plan.
