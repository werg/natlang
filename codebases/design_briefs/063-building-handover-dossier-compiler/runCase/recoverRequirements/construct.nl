---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHandoverRequirement
---
Construct a HandoverRequirement draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read the supplied handover specification, asset schedules, and contractual requirements to identify required documents, tests, training, and approvals. Separate documentary requirements from technical sign-off and preserve conditions specific to asset classes or building areas.
