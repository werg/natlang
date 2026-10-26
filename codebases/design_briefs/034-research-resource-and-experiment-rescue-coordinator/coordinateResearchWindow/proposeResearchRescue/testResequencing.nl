---
args: { mandates: ProtocolMandate[], assignments: AssignmentSearch, resources: ResourceAssessment[] }
returns: RescueAlternative[]
---
For unscheduled mandates search order changes that preserve every protocol prerequisite and fit the same sample-use interval. Never move a consuming experiment ahead of its required preparation. Return the changed ordering and affected experiment IDs as a proposal with no protocol deviation when all hard conditions remain true.
