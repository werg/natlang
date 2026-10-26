---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: GardenAgreement[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for coordinateSeason. Turn accepted plans into bounded seasonal tasks, shared-tool reservations, and reminders. Adapt to changed availability or observations, preserve agreements across seasons, and record actual completion without treating missed tasks as automatic grounds for exclusion. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
