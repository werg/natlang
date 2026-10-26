---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDebtOutcome
---
Construct a DebtOutcome draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare post-change workflow evidence with the original problem definition, accounting for changes in workload and observation. Distinguish successful delivery from actual improvement and reopen mechanisms that the completed work did not address.
