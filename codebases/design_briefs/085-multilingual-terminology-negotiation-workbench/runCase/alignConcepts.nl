---
description: alignConcepts domain stage for Multilingual Terminology Negotiation Workbench.
args:
  goal: string
  records: TermOccurrence[]
  evidence: EvidenceItem[]
returns: StageResultMultilingualConcept
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMultilingualConcept values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare conceptual boundaries across languages and teams, identifying exact, broader, narrower, and partial equivalence. Preserve concepts that do not map cleanly and avoid forcing one language’s categorization onto all others. The stage gate is: Concept alignment is supported by definitions and counterexamples, not dictionary matching alone.
