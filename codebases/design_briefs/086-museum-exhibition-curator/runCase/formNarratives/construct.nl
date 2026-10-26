---
description: Construct a ExhibitionArgument draft for formNarratives.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CuratorialEvidence[] }
returns: DraftExhibitionArgument
---
Assemble the ExhibitionArgument as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Narrative claims remain supported and contested interpretations are represented fairly.
