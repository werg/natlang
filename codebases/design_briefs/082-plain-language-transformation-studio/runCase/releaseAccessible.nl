---
description: releaseAccessible domain stage for Plain Language Transformation Studio.
args:
  goal: string
  records: ComprehensionProbe[]
  evidence: EvidenceItem[]
returns: StageResultPlainLanguageRelease
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPlainLanguageRelease values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconcile reviewer feedback, maintain side-by-side source mapping, and publish the accessible edition with its scope and version. Revalidate affected meaning units after source changes and preserve access to authoritative wording where required. The stage gate is: Independent fidelity review passes; readability alone is not the acceptance criterion.
