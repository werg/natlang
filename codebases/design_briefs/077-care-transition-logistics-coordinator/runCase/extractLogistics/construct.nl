---
description: Construct a TransitionRequirement draft for extractLogistics.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CareTransition[] }
returns: DraftTransitionRequirement
---
Assemble the TransitionRequirement as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No generated change to clinical content or priority.
