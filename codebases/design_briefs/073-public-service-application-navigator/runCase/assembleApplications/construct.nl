---
description: Construct a ApplicationTask draft for assembleApplications.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: NavigationQuestion[] }
returns: DraftApplicationTask
---
Assemble the ApplicationTask as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Application content is evidence-backed and explicitly confirmed by the applicant where required.
