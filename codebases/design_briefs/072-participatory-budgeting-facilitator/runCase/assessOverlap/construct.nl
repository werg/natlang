---
description: Construct a ProposalRelationship draft for assessOverlap.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CivicProposal[] }
returns: DraftProposalRelationship
---
Assemble the ProposalRelationship as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Merging preserves distinct intents and does not erase minority proposals.
