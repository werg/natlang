---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftObservationQuestion
---
Construct a ObservationQuestion draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Form targeted follow-up requests that could distinguish plausible interpretations or correct metadata. Use respectful wording, avoid leading the observer toward the expected label, and route specialist-sensitive cases for expert review.
