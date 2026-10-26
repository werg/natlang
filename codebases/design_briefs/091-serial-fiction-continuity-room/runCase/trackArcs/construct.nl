---
description: Construct a StoryObligation draft for trackArcs.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CanonFact[] }
returns: DraftStoryObligation
---
Assemble the StoryObligation as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Arc claims are grounded in the actual text or approved author intent.
