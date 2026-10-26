---
args: { candidates: TransferFinding[], checkpoint: LearnerCheckpoint, input: LearnerSessionInput }
returns: TransferFinding[]
---
Join each transfer task with hint exposure and scaffold history. Mark any response after a conceptual hint as assisted, and never use it alone to establish mastery.
