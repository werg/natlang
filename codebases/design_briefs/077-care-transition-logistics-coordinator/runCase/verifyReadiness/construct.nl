---
description: Construct a TransitionReadiness draft for verifyReadiness.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TransitionOption[] }
returns: DraftTransitionReadiness
---
Assemble the TransitionReadiness as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No transition-ready claim while a required logistical prerequisite is unverified.
