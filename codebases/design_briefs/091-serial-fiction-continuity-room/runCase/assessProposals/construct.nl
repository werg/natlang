---
description: Construct a StoryProposalFinding draft for assessProposals.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: StoryObligation[] }
returns: DraftStoryProposalFinding
---
Assemble the StoryProposalFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No unexplained violation of established knowledge or chronology.
