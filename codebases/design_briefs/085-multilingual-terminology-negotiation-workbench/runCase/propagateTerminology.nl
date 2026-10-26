---
description: propagateTerminology domain stage for Multilingual Terminology Negotiation Workbench.
args:
  goal: string
  records: TerminologyDecision[]
  evidence: EvidenceItem[]
returns: StageResultTerminologyImpact
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTerminologyImpact values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Identify content using the affected concept and sense, propose context-aware revisions, and preserve unrelated homographs. Revalidate grammar and meaning after replacement and version the glossary and dependent release artifacts together. The stage gate is: No global search-and-replace across different senses; affected content passes fidelity checks.
