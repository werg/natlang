---
args: { findings: TransferFinding[], input: LearnerSessionInput }
returns: TransferFinding[]
---
Update the learner record only from current authorized evidence satisfying the competency standard. Preserve uncertainty and instructor review when independent transfer is not established; do not infer durable ability from a single prompted response.
