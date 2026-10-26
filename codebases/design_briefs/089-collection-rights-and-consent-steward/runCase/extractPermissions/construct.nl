---
description: Construct a CollectionPermission draft for extractPermissions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CollectionUse[] }
returns: DraftCollectionPermission
---
Assemble the CollectionPermission as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Absence of a restriction record is not treated as permission.
