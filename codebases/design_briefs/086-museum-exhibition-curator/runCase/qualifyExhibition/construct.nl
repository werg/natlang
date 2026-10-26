---
description: Construct a ExhibitionRelease draft for qualifyExhibition.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ExhibitionLabel[] }
returns: DraftExhibitionRelease
---
Assemble the ExhibitionRelease as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: All required specialist approvals and exact physical constraints pass before installation.
