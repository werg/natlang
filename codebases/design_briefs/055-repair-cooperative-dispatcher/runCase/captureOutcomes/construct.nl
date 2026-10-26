---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRepairOutcome
---
Construct a RepairOutcome draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare post-work checks and customer observations with the original symptoms. Record partial success, recurrence, and specialist referrals, then extract scoped lessons that improve future routing without treating one repair as universal proof of a cause.
