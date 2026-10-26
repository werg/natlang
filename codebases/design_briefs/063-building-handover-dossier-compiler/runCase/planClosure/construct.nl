---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHandoverClosureTask
---
Construct a HandoverClosureTask draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Group gaps into actionable requests for contractors, inspectors, or facilities staff. Prioritize dependencies affecting safe or practical use under supplied rules, and preserve accepted temporary arrangements with owners and expiry rather than hiding them as complete.
