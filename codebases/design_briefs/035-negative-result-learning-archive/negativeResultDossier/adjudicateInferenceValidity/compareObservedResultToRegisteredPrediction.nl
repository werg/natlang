---
args: { attempt: AttemptTimeline, snapshot: AttemptArchiveCorpus }
returns: InferentialStatus
---
Compare the valid measured result with the prediction registered before execution and the exact tested conditions. Return valid-negative only when the measurement contradicts that prediction; otherwise preserve positive or indeterminate findings as uncertain, not negative.
