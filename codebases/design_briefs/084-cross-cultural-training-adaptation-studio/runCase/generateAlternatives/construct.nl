---
description: Construct a AdaptedScenario draft for generateAlternatives.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ContextFitFinding[] }
returns: DraftAdaptedScenario
---
Assemble the AdaptedScenario as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Instructional equivalence and contextual plausibility are separately reviewed.
