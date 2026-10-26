---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftChangeVerification
---
Construct a ChangeVerification draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare actual implemented configurations with approved applicability and acceptance criteria. Preserve authorized exceptions and unresolved field cases, then close the change only when required consumers and records reflect the correct revision state.
