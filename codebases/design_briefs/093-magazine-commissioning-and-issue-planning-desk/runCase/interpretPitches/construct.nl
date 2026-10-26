---
description: Construct a EditorialPitch draft for interpretPitches.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MagazineIssue[] }
returns: DraftEditorialPitch
---
Assemble the EditorialPitch as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Pitch evaluation is grounded in the supplied editorial goals and feasible reporting.
