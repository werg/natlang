---
description: Construct a StoryEdition draft for verifyEdition.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: StoryRevision[] }
returns: DraftStoryEdition
---
Assemble the StoryEdition as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Global continuity and approved narrative constraints pass; creative quality receives separate editorial review.
