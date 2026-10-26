---
description: Construct a TheatreScene draft for modelDramaturgy.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ImmersiveProduction[] }
returns: DraftTheatreScene
---
Assemble the TheatreScene as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Scene transitions respect knowledge and narrative prerequisites.
