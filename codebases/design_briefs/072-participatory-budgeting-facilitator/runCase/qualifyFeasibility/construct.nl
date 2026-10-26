---
description: Construct a ProposalFeasibility draft for qualifyFeasibility.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ProposalRelationship[] }
returns: DraftProposalFeasibility
---
Assemble the ProposalFeasibility as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Eligibility and feasibility decisions cite supplied rules and verified estimates.
