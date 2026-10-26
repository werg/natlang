---
description: recoverBrandConstraints domain stage for Brand And Design Language Studio.
args:
  goal: string
  records: BrandProject[]
  evidence: EvidenceItem[]
returns: StageResultBrandConstraint
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftBrandConstraint values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Translate the client brief, product facts, and existing assets into desired associations and hard constraints. Separate aspirational positioning from demonstrable capability, identify legal or rights review needs, and preserve audience guidance without inferring personal traits. The stage gate is: Product claims and prohibitions have authoritative sources.
