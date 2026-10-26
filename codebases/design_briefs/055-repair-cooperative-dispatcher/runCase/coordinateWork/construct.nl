---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRepairWorkPlan
---
Construct a RepairWorkPlan draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Sequence inspection, parts procurement, repair, and verification under approved procedures. Reserve shared tools, obtain consent for cost or scope changes, and preserve an explicit stop-and-escalate path when findings exceed the original plan.
