---
args: { attempts: AttemptTimeline[], snapshot: AttemptArchiveCorpus }
returns: InferentialStatus[]
---
For each attempt bind completion = testCompletion(attempt), integrity = testMeasurementIntegrity(attempt, snapshot), and outcome = compareObservedResultToRegisteredPrediction(attempt, snapshot). Merge those three typed results into one inferential status for that attempt. A missing procedure step is incomplete; a failed measurement chain is measurement-invalid; only valid completion and measurement plus an outcome testing the registered prediction may be valid-negative. Preserve uncertainty when evidence cannot choose among these states.
