---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRepairAssignment
---
Construct a RepairAssignment draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Match tasks to volunteers or specialists using stated skills, verified qualifications where required, availability, and consent. Do not infer capability from identity or enthusiasm, and preserve supervision requirements for trainees.
