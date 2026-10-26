---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftGardenTask
---
Construct a GardenTask draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Turn accepted plans into bounded seasonal tasks, shared-tool reservations, and reminders. Adapt to changed availability or observations, preserve agreements across seasons, and record actual completion without treating missed tasks as automatic grounds for exclusion.
