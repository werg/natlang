---
args: { variant: WorkfloorInstructionDraft, restarts: RestartCondition[], exceptions: LocalException[] }
returns: ReplayFinding
---
For every interrupted step, replay the transition from the recorded interruption state and require each isolation and repeat check before resuming. Verify every exception leaves its named inspection guard intact. List exact failed steps and the instruction fragments requiring revision.
