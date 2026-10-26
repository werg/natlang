---
args: { archiveRequest: ArchiveRequest, attemptStore: AttemptArchiveCorpus }
returns: AttemptTimeline[]
---
1. Bind plannedSteps = extractIntendedProcedure(attemptStore.attempts, attemptStore.evidencePages).
2. Bind performedSteps = orderExecutionEvents(plannedSteps, attemptStore.evidencePages), retaining missing, interrupted, and out-of-order events.
3. Bind outcomeChain = validateMeasurementChain(performedSteps, attemptStore.evidencePages). Separate raw observation, instrument health, data loading, and post-processing provenance.
4. Return one AttemptTimeline for each attempt with source-bounded intended and actual steps, omissions, deviations, measurements, and explicit evidence references.
