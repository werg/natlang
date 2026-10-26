---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftObservationInterpretation
---
Construct a ObservationInterpretation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret local names and descriptive phrases against project terminology and context. Retain multiple plausible identities when evidence is insufficient, and avoid forcing unusual observations into familiar categories merely to complete the dataset.
