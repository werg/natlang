---
description: Construct a CivicProposal draft for interpretProposals.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: BudgetRound[] }
returns: DraftCivicProposal
---
Assemble the CivicProposal as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Proposal normalization does not silently change the project’s purpose.
