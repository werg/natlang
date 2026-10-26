---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPracticeComplication
---
Construct a PracticeComplication draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Select a complication appropriate to the learner’s progress and learning objective. Ensure it changes a meaningful decision, preserves a safe recovery path, and does not retroactively invalidate an action that was correct under earlier visible facts.
