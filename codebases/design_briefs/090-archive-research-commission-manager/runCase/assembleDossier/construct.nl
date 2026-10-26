---
description: Construct a ResearchDossier draft for assembleDossier.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ArchiveFinding[] }
returns: DraftResearchDossier
---
Assemble the ResearchDossier as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Another researcher can reproduce the searched scope and distinguish evidence from inference.
