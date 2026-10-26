---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReuseGoal
---
Construct a ReuseGoal draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Translate stakeholder objectives into required functions and explicit tradeoffs. Separate mandatory supplied constraints from preferences, identify incompatible goals, and avoid choosing political or commercial priorities not authorized by the project brief.
