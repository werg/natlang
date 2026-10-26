---
args: { mandates: ProtocolMandate[], resources: ResourceAssessment[] }
returns: AssignmentFrontier
---
Freeze all mandate IDs, eligible instrument choices, sample-window endpoints, duration, required capability, operator constraint, and source refs into the initial frontier. Start deferredExperimentIds empty. Do not generate a candidate for an experiment with no eligible resource. Keep that experiment pending with its exact resource blocker so the search can still schedule independent experiments; the frontier must later retain a distinct deferred branch when it expands that experiment.
