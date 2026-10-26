---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftItemChallenge
---
Construct a ItemChallenge draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Independently solve each item, attempt reasonable alternate readings, and test rubric consistency on borderline responses. Repair items and rubrics together, then rerun coverage and duration checks before releasing a frozen assessment edition.
