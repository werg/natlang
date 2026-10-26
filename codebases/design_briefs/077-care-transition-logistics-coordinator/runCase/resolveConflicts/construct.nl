---
description: Construct a TransitionOption draft for resolveConflicts.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TransitionTask[] }
returns: DraftTransitionOption
---
Assemble the TransitionOption as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Administrative replanning never substitutes for clinical approval.
