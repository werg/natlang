---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftTestFailureFinding
---
Construct a TestFailureFinding draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare failures with requirements, execution traces, and fixture health. Distinguish a product defect from a bad oracle, flaky setup, or unavailable dependency; propose the smallest repair without weakening a valid assertion to match defective behavior.
