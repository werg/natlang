---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftBehavioralChange
---
Construct a BehavioralChange draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read release diffs, requirements, and observed behavior to identify changed semantics, defaults, timing, and error handling. Distinguish internal refactoring from externally meaningful changes even when schemas and type signatures remain compatible.
