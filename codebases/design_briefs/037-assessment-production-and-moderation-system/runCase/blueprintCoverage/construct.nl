---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAssessmentBlueprint
---
Construct a AssessmentBlueprint draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Translate the assessment purpose into observable competencies and a balanced task blueprint. Distinguish intended skills from incidental reading or tool demands, and identify constraints on duration, accessibility, and allowable assistance.
