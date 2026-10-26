---
description: Construct a StoryRevision draft for generateRepairs.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: StoryProposalFinding[] }
returns: DraftStoryRevision
---
Assemble the StoryRevision as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Repairs satisfy hard canon constraints unless a reviewed canon change is explicitly proposed.
