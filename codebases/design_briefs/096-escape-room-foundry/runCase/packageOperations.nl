---
description: packageOperations domain stage for Escape Room Foundry.
args:
  goal: string
  records: PuzzlePlaytest[]
  evidence: EvidenceItem[]
returns: StageResultEscapeOperations
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEscapeOperations values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Produce the full operating pack: setup, reset, hints, accessibility variants, prop checks, and staff recovery procedures. Verify reset completeness and session isolation, then obtain safety and creative review separately from the mathematical solvability check. The stage gate is: Independent solvability, safe operation, reliable reset, and human enjoyment review all required.
