# Conversation continuation checkpoints

Long model invocations are divided into short conversation segments. After six
work turns, when the task remains incomplete and the latest result is durable,
the model receives a no-tools request for a working note. The note is saved on
the lambda. The next request is built from the original instructions, current
line marks, workspace, recent effects, and note. No earlier chat messages are
copied into the next segment. A checkpoint does not end or quiesce the lambda.

The boundary waits after `read` and `run_code`, because their results may be
needed by the next action without having been written into the workspace. It
also waits after a rejected action or a nudge. When the return and marks are
complete, the agent can finish normally without a checkpoint. The six-turn
setting is a rollover point, not a task budget or failure condition.

Synthetic reference turns restart from workspace state on the same boundary.
They do not fabricate note targets. Teacher captures retain checkpoint turns,
including model reasoning and the exact pre-checkpoint conversation. The
teacher trajectory IR and replay bridge recognize those turns; older trajectories
replay with checkpoints disabled. Audit traces retain full history, while SFT
examples after a checkpoint contain only the new segment.

Data generation and application evaluation were paused on 2026-09-20 while this
change is checked. Existing raw data and journals remain available. The
existing synthetic SFT files still contain full conversation prefixes; they
need a fresh materialization from frozen programs before a continuation-focused
training run. Do not splice rendered SFT text without reconstructing runtime
state. Bonsai may be used for targeted probes while collectors remain paused.

Local checks: 369 tests passed, one skipped. A Bonsai probe
completed a partial-record task after a fresh checkpoint prompt with four
messages. A one-turn boundary initially caused unnecessary repeated actions
after a complete return; completion detection and transient-result handling
were added before the successful rerun. A full teacher corpus comparison and
training throughput measurement remain to be done before restarting collection.
