---
args: { previous: AnalysisSlice | null, currentRubricVersion: string, changedEvidenceIds: ID[] }
returns: RubricReclassificationPlan
---
Keep the previous run immutable with its rubric version. Select only rows whose judgment depended on the changed rubric and create a new run version for them. Reuse unaffected qualified row judgments only when the rubric and source records are unchanged.
