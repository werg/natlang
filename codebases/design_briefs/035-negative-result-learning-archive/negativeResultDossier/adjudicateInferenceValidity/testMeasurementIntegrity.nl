---
args: { attempt: AttemptTimeline, snapshot: AttemptArchiveCorpus }
returns: InferentialStatus
---
Require valid calibration, control behavior, complete acquisition, and faithful data loading/transformation. If any required link failed, classify the result as measurement-invalid and name the broken link; do not treat the measured value as evidence against the hypothesis.
