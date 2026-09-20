# Conversation continuation checkpoints

Long model invocations are divided into short conversation segments. After 12
conversation messages or six work turns, when the task remains incomplete and the latest result is durable,
the model receives a no-tools request for a working note. The note is saved on
the lambda. The next request is built from the original instructions, current
line marks, workspace, recent effects, and note. No earlier chat messages are
copied into the next segment. A checkpoint does not end or quiesce the lambda.

The threshold counts conversation items, not tokens. A single multi-call turn
can exceed it; it is a rollover point rather than a hard cap. The boundary waits after `read` and `run_code`, because their results may be
needed by the next action without having been written into the workspace. It
also waits after a rejected action or a nudge. When the return and marks are
complete, the agent can finish normally without a checkpoint. The six-turn
setting is a rollover point, not a task budget or failure condition.

Synthetic reference turns restart from workspace state on the same boundary.
They do not fabricate note targets. Teacher captures retain checkpoint turns,
including model reasoning and the exact pre-checkpoint conversation. The
teacher trajectory IR records `note: {text, author: "teacher"}` and the message
threshold on each checkpoint. Older trajectories have no model-authored notes
and replay with checkpoints disabled. Audit traces retain full history, while SFT
examples after a checkpoint contain only the new segment.

Data generation and application evaluation were paused on 2026-09-20 while this
change is checked. Existing raw data and journals remain available. The
older synthetic SFT files still contain full conversation prefixes; the new
continuation bundle below replaces them for a continuation-focused training
run. Do not splice rendered SFT text without reconstructing runtime state.
Bonsai may be used for targeted probes while collectors remain paused.

Local checks: 371 tests passed, one skipped. A Bonsai probe
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
capture/replay test covers this path. The TS host port passes 39 native parity
tests against Python, including continuation, note persistence, and effect
journal visibility.

The TypeScript host and browser host now use the same message/turn rollover, persist
`continuation_note` in lambda values, expose `args@effects`, and include
the note and recent effects in the fresh workspace opening. Paired fixtures
exercise checkpointing, value round trips, and tool-surface parity across both
runtimes.

## Existing training data decision (2026-09-20)

Keep accepted teacher trajectories and replay them without inserting synthetic
checkpoints. A replay reset would either invent a note target or change the
teacher's observed context. Recollect targeted examples later if the teacher
fails with the new checkpoint protocol; do not bulk regenerate the reviewed
teacher corpus. In the rendered 921-row reviewed bundle, 917 prompts have at
most 12 preceding conversation items. The four longer prompts come from two
trajectories: three turns in one leaf's read/read/reply tail (14, 16, 18 items)
and one program reply (14 items). The read outputs are transient and cannot be
discarded at a safe boundary. This is a small, explicit exception to the
predominantly short-history policy.

The older 280,472-row synthetic bundle has 42,669 prompts above 12 items
(15.2%, maximum 57). Regenerate it from the frozen 10,000-program IR with the
message-count rollover before the next training run; keep the old bundle only
as a reproducible baseline. Reference-agent restarts are state-derived and have
no teacher note. Do not splice rendered prompts or impose a token cutoff.

A 100-program replay pilot produced 2,664 turns, of which 2,653 have at most
12 context items; the maximum is 14. All programs verified. The new SFT
exporter records `context_items` directly from structured messages, so the
next rendered bundle can be audited without depending on template delimiters.

The full frozen 10,000-program replay verified all programs and yielded the
same 280,472 targets. The rendered `data/synthetic-continuation-items.sft.jsonl`
was checked against its manifest and has 279,385 (99.61%) prompts with at
most 12 context items, 1,087 with 13 or 14, and none above 14. The message
threshold is soft because a single durable action can add multiple feedback
items before the next safe boundary. The local LFM2.5-350M rendering used the
same template hash as the prior synthetic bundle.

One targeted Bonsai recollection of frozen `73:map_then_count:13` was admitted.
Its 22-turn IR has one teacher-authored checkpoint note with the 12-item and
two-turn settings recorded. Replay preserved the result, materialized the note
as a target, and reopened the next sample from four messages containing the
note and current workspace. Rendered with the same student template, this
trajectory is included in `data/teacher-reviewed-with-continuation.sft.jsonl`
(943 pairs total). Bulk teacher generation was subsequently resumed on
2026-09-20. The balanced whole-program collector writes complete teacher
choices, reasoning, tool calls, and checkpoints to
`runs/teacher-program-balanced-s909-pass1.ir.jsonl`; the page-content leaf
collector writes review and trajectory IR to
`runs/teacher-leaves-s74-pages-20260920*`. These are append-only collection
outputs; accepted trajectories must still pass replay before SFT export.

The original seed-74 frozen IR is a historical snapshot. Its shopkeeper
effect contracts are stale for five programs (IDs 3787, 8767, 8803, 8811,
8824): replay emits the correct sequence but the saved contract contains too
few effects. Rebuilding from the pinned generator manifest with current sources
produced `data/external_pilot/synthetic-s74-refrozen-20260920.ir.jsonl`.
Its manifest explicitly records source and reference-bank drift, so this is a
new corpus revision rather than a historical match. All 10,000 records pass
the IR schema audit, and the five old shopkeeper failures pass individual
runtime verification. Its initial 179 provisional programs remain excluded
from SFT until leaf references are admitted. Full continuation replay verified
all 10,000 programs, yielding 279,335 turns: 271,832 eligible and 7,503
provisional. SFT rendering uses this new revision; a separate watcher refreshes
it after the page-content collector finishes. The teacher replay bridge has
no episode cap; the finite recorded trajectory itself bounds replay.
