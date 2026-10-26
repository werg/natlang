---
description: Construct a ResearchCommissionScope draft for frameCommission.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ArchiveCommission[] }
returns: DraftResearchCommissionScope
---
Assemble the ResearchCommissionScope as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Each subquestion has a plausible evidence path and an explicit inferential limit.
