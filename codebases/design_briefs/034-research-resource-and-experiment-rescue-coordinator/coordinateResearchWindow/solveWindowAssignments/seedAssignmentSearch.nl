---
args: { mandates: ProtocolMandate[], resources: ResourceAssessment[] }
returns: AssignmentFrontier
---
Freeze all mandate IDs, eligible instrument choices, sample-window endpoints, duration, required capability, operator constraint, and source refs into the initial frontier. Do not generate a candidate for an experiment with no eligible resource. Record that experiment as pending with its exact resource blocker so the search can still schedule independent experiments.
