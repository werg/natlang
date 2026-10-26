---
description: Construct a CivicPortfolio draft for constructPortfolios.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ProposalFeasibility[] }
returns: DraftCivicPortfolio
---
Assemble the CivicPortfolio as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact budget feasibility and complete disclosure of exclusions and dependencies.
