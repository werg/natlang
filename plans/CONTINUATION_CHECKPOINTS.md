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

Local checks: 370 tests passed, one skipped. A Bonsai probe
completed a partial-record task after a fresh checkpoint prompt with four
messages. A one-turn boundary initially caused unnecessary repeated actions
after a complete return; completion detection and transient-result handling
were added before the successful rerun. A full teacher corpus comparison and
training throughput measurement remain to be done before restarting collection.

## Small paired Bonsai probe

`scripts/compare_continuations.py` runs one case at a time with a 360-second
probe deadline. Both conditions use the same frozen program, seed, teacher
prompt, decoder settings, and machine. Each run writes raw turns and a trace
under `runs/continuation-compare-*`; it does not add training data.

| Case | Rollover | Correct | Requests | Prompt tokens | Completion tokens | Max prompt | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Partial record | off | yes | 5 | 9,509 | 349 | 2,056 | 38.16 |
| Partial record | 2 | yes | 5 | 9,509 | 349 | 2,056 | 39.37 |
| Map then count | off | yes | 21 | 41,863 | 2,215 | 2,941 | 269.72 |
| Map then count | 2 | yes | 22 | 42,577 | 2,562 | 2,868 | 290.72 |

The partial record had no safe boundary: its read and code results were
transient, and the first durable write completed the task. Map then count had
one checkpoint in its root invocation. A two-turn rollover was too early to
save tokens there: it reduced the largest prompt by 73 tokens but added 714
prompt and 347 completion tokens overall. This comparison does not establish
throughput gains for longer single-lambda histories. The normal six-turn
rollover did not fire in these two programs.

The map checkpoint's 256-token note response hit its length limit and ended
mid-sentence. Raising the note allowance to 512 in Python and TS let the same
checkpoint prompt finish normally in 323 tokens; the note remains truncated
to 800 characters before storage.

Whole-program teacher capture now retains the checkpoint phase, setting,
context, and offered tools, allowing the replay bridge to materialize the
checkpoint note and the following fresh-context turn. A scripted end-to-end
capture/replay test covers this path. The TS host port passes 38 native parity
tests against Python, including continuation, note persistence, and effect
journal visibility.

The TypeScript host and browser host now use the same six-turn rollover, persist
`continuation_note` in lambda values, expose `args@effects`, and include
the note and recent effects in the fresh workspace opening. Paired fixtures
exercise checkpointing, value round trips, and tool-surface parity across both
runtimes.
