---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDiagnosticAction
---
Construct a DiagnosticAction draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Choose from approved read-only or bounded diagnostic operations, considering expected discrimination and operational risk. Validate scope and rate limits, request commander approval for intrusive checks, and record what each possible result would change.
