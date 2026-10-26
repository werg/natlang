---
description: Construct a AdaptationTest draft for replayLearning.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AdaptedScenario[] }
returns: DraftAdaptationTest
---
Assemble the AdaptationTest as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: The same competency distinctions remain observable in both versions.
