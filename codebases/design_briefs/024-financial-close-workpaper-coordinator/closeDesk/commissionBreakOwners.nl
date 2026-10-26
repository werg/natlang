---
args: { request: CloseCycleRequest; breaks: CloseBreakLedger; transfers: EvidencePage[] }
returns: CloseTaskQueue
---
Create one investigation task for each material break with an unresolved fact. Match any preparer-transfer receipt to the original owner and named successor; preserve the former authorship and leave the task unassigned if successor authority is missing. State the exact evidence that would resolve the break and its close deadline from supplied records. Do not mark a task complete from reassignment. Return the queue, owner gaps and evidence refs.