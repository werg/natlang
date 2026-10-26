---
description: Construct a ArchiveLead draft for followEvidence.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ArchiveSearchPlan[] }
returns: DraftArchiveLead
---
Assemble the ArchiveLead as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every lead has a question-specific rationale and source provenance.
