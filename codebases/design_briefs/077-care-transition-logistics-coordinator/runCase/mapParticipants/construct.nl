---
description: Construct a TransitionParticipant draft for mapParticipants.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TransitionRequirement[] }
returns: DraftTransitionParticipant
---
Assemble the TransitionParticipant as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every responsibility has explicit acceptance and permitted information scope.
