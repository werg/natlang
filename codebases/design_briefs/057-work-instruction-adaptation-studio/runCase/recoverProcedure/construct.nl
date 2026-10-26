---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftProcedureRequirement
---
Construct a ProcedureRequirement draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Parse the master instruction into actions, guards, prerequisites, and verification obligations. Distinguish explanatory advice from mandatory controls, preserve warnings in their operative context, and identify cross-references needed to execute the procedure correctly.
