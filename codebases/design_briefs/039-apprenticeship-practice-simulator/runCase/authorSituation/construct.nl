---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPracticeScenario
---
Construct a PracticeScenario draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Construct a realistic situation requiring the supplied procedure and meaningful information gathering. Separate facts visible at the start from facts revealed by appropriate questions, and include complications that test judgment without requiring unauthorized actions.
