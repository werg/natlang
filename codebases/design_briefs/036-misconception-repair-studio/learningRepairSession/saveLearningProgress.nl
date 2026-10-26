---
args: { previous: LearnerCheckpoint, observations: LearnerEvidence[], hypotheses: HypothesisSet[], interventions: LearningIntervention[], transfer: TransferFinding[], handoff: BoundedHandoff }
returns: LearnerCheckpoint
---
Persist the exact last completed task and its version, previous hint exposure, completed accommodations, unresolved question IDs, and handoff count. Store private hypothesis IDs outside learner-visible text. Do not save a proposed intervention as a completed task or commit a mastery update without an authorized host write.
