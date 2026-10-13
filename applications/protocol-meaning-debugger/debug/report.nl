---
description: Return diagnosis, exact experiment receipts and bounded repair evidence.
args:
  context: InvestigationContext
  state: InvestigationState
  diagnosis: Diagnosis
  repair: RepairReplay | null
returns: DebugReport
---
1. Preserve baseline from context, original contrasts, final hypotheses and all consumed probe outcomes.
   Do not reconstruct a simpler explanation from raw files after child stages return.
2. Include diagnosis.rootCauses, coincidences and unresolved unchanged. A supported cause names
   the selected definition, relevant contrast and probe evidence actually used to localize it.
3. Include the full RepairReplay when present: proposed spans, rebound plans, probe receipts and
   finding coverage. Null repair means no verified proposal, not a no-op successful migration.
4. Collect witnesses from original interpretations, contrasts, diagnoses and proposed edits. Call
   measurementLab.checkWitnesses against original documents for original-source witnesses.
5. Repaired-source witnesses belong to the edited snapshot and its audit exact checks. Do not verify
   a replacement quotation against original text or silently change the cited source revision.
6. Combine interpretation unknowns, lineage defects, experiment uncertainty, diagnosis unresolved
   and repair findings into unresolved. Deduplicate identical meanings without losing their source IDs.
7. Return caseId, baseline, contrasts, hypotheses, outcomes, diagnosis, repair, unresolved and sourceChecks.
   Exact numerical receipts retain numerator, denominator and selectedReadingIds, including null for0attempts.
8. Completion requires coherent source-bound meanings, discriminating interventions and replayed repair.
   An external checker can verify exact invariants, while witness entailment and real literal consumption
   still need independent semantic and live trace review.
