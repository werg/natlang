---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftLearningIntervention
---
Construct a LearningIntervention draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Select a suitable explanation and contrastive examples, then create practice with gradually reduced scaffolding. Adapt to observed responses without simply repeating the original answer, and escalate persistent ambiguity to the instructor.
