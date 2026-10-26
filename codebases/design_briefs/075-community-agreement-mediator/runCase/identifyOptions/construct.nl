---
description: Construct a CommunityOption draft for identifyOptions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MediationPosition[] }
returns: DraftCommunityOption
---
Assemble the CommunityOption as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Options are feasible and accurately represent who would give up what.
