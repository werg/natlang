---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftChangeApplicability
---
Construct a ChangeApplicability draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret effective serials, lots, dates, configurations, and authorized exceptions. Distinguish new production from retrofit obligations and preserve fielded units outside scope rather than applying the latest revision universally.
