---
args: { results: StageOutputs, questions: QuestionBatch }
returns: ReviewPacket
---
1. Present supported decisions, alternatives, action scope, expected consequences, and immutable approval bindings.
2. Explain supported decisions, unresolved alternatives, action scope, and expected consequences. Group approvals only when they share authority and dependencies. Separate reviewer judgment from the executor’s recommendation and bind approval requests to immutable hashes.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
