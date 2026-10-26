---
description: Construct a TheatreCuePlan draft for coordinatePerformance.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: SceneVariant[] }
returns: DraftTheatreCuePlan
---
Assemble the TheatreCuePlan as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No conflicting resource use, impossible timing, or unauthorized safety-critical cue.
