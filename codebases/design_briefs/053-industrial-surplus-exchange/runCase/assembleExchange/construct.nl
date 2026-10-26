---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftExchangePackage
---
Construct a ExchangePackage draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Construct feasible multi-party exchanges using exact quantity, reservation, and transport services. Account for dependent legs, storage windows, and adaptation costs, and preserve unallocated surplus and unmet needs explicitly.
