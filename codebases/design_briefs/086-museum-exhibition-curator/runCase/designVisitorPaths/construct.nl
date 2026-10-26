---
description: Construct a ExhibitionZone draft for designVisitorPaths.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ExhibitionArgument[] }
returns: DraftExhibitionZone
---
Assemble the ExhibitionZone as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Physical layout and artifact constraints are feasible under supplied venue data.
