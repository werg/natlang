---
description: Construct a ReferralRecordMatch draft for matchPatientRecords.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ReferralRequirement[] }
returns: DraftReferralRecordMatch
---
Assemble the ReferralRecordMatch as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No cross-patient attachment or unsupported identity merge.
