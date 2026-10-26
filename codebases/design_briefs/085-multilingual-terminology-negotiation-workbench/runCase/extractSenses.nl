---
description: extractSenses domain stage for Multilingual Terminology Negotiation Workbench.
args:
  goal: string
  records: TerminologyProject[]
  evidence: EvidenceItem[]
returns: StageResultTermOccurrence
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTermOccurrence values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Collect terminology in actual use across expert documents and translations, preserving grammatical and domain context. Separate different senses of the same word and identify cases where different words denote one concept rather than assuming one-to-one lexical equivalence. The stage gate is: Every sense has contextual evidence.
