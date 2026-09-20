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
change was checked, then teacher collection resumed at the user's direction.
Existing raw data and journals remain available. The
older synthetic SFT files still contain full conversation prefixes; the new
continuation bundle below replaces them for a continuation-focused training
run. Do not splice rendered SFT text without reconstructing runtime state.

Local checks: 371 tests passed, one skipped. A Bonsai probe
completed a partial-record task after a fresh checkpoint prompt with four
messages. A one-turn boundary initially caused unnecessary repeated actions
after a complete return; completion detection and transient-result handling
were added before the successful rerun. A full teacher corpus comparison and
training throughput measurement remain open while collection proceeds.

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
An early snapshot of the first ten page-content attempts had nine accepted
trajectories. Their 23 replayed turns rendered with the LFM2.5-350M template
and were combined with the reviewed teacher bundle in
`data/teacher-available-20260920.sft.jsonl` (966 pairs). This is a fixed
snapshot; later collector rows require a new replay and bundle.
After the page-content pass, a queued `say` pass collects the remaining
shopkeeper dialogue cases in `runs/teacher-leaves-s74-say-20260920*` for manual
semantic review; it does not auto-admit them to the reference bank.

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
provisional. All 271,832 eligible turns rendered to
`data/synthetic-s74-refrozen-continuation.sft.jsonl`. The manifest count and
unique ID count agree; 270,688 prompts (99.58%) have at most 12 conversation
items, and the maximum is 14. Its LFM2.5-350M template hash is
`70278c3c69a31e89c2383bb2c4cb5f22ec8456069bcd194d553c30a00dbe1b05`.
A separate watcher refreshes the IR, verifies all programs, and renders a new
SFT bundle after the page-content collector finishes. It checks the pair count
and the pinned student template hash before publishing the rendered bundle.
The teacher replay bridge has no episode cap; the finite recorded trajectory
itself bounds replay.

## Collection status (2026-09-21)

The five superseded rendered synthetic SFT snapshots were removed on
2026-09-21 after the archive audit; the temporary compressed copies were also
removed. Older v1–v6 reference outputs, v5–v8 student SFT exports, pre-IR
external traces and SFT renders, quarantined s112 support data, obsolete
agent-behavior outputs, and historical pilot model runs and exports were
removed in the same cleanup. Paths to these historical outputs elsewhere in
this document are provenance, not available training inputs. The current IR,
manifests, original external datasets, reference reviews, and active teacher
trajectories remain available for the corrected rebuild.

The page-content pass completed 110 attempts: 100 accepted and admitted,
eight quiesced, and two raised exceptions. Its reference-bank refresh verified
all 10,000 seed-74 programs and rendered 274,406 eligible SFT pairs in
`data/external_pilot/synthetic-s74-refrozen-after-pages-1a566fa5eb67.sft.jsonl`.
The other 106 programs remain provisional. All 100 admitted page trajectories
replayed into 240 teacher turns.

The dialogue pass completed 76 attempts. Its judge accepted 63, but none were
automatically admitted. Manual inspection found accepted `sell_list_price`
outputs that merely quote or offer a price even though the action says a sale
has already happened; the dialogue pass needs semantic review and likely a
targeted rerun before use as gold.

The balanced whole-program collector remains active. A fixed snapshot of its
first 15 completed programs had seven accepted trajectories, which replayed
into 146 turns. `data/teacher-available-s74-current.sft.jsonl` combines those
with the reviewed teacher bundle and the 240 page turns: 1,329 pairs total.

### Post-crash review (2026-09-21)

An unexpected reboot interrupted the first whole-program pass during row 15.
The 15 completed rows survived; Bonsai and its watchdog were restarted, and
rows 15–23 resumed in `runs/teacher-program-balanced-s909-pass2.ir.jsonl`.
The prior boot's kernel journal has AMD display-driver warnings and a display
flip timeout immediately before the reboot; this is evidence of a display
failure, not a confirmed cause. The collector now supports `--resume`, checks
completed source rows and model identity, and preserves interrupted trace files.

Manual semantic review admitted 40 dialogue references and replayed their full
teacher trajectories into 108 turns. The current reviewed teacher bundle is
`data/teacher-available-s74-reviewed.sft.jsonl` (1,437 pairs). The review and
judge overrides are pinned in `data/say-s74-review-20260921.json`. A second
review of the original reference bank found 80 earlier `sell_*` lines that
quote a price, offer a later transfer, or otherwise fail to say that the sale
is complete. The pinned review is in
`data/say-original-bank-review-20260921.json`; a reversible bank prune is
queued after the active page retry so it cannot race the reference writer.

The page retries finished and the 80 reviewed bad sale references were pruned
with an exact backup at `data/leaf_references-pre-s74-prune-20260921.jsonl`.
The new `say` pass is collecting the 36 originally missing keys without auto
admission. The pruned keys were embedded as concrete oracle cases in the old
frozen IR, so a template-only collector would silently miss them.
`teacher_leaves.py --reopen-removed-from` now finds those keys from the bank
backup and revisits their original frozen arguments. Seed 73 covers 79 of the
80 keys, and seed 74 covers the last; collection is queued after the current
36-key pass and still requires manual semantic review. A separate watcher
will replay and render the nine remaining whole-program attempts after their
collector finishes. Rebuild the two synthetic seeds only after the reviewed
new references are admitted.

**Do not train from the existing seed-73 or seed-74 synthetic SFT snapshots**
until they are rebuilt. Their actual source IRs embed these questionable sale
references in 125 and 99 programs, respectively. The page retry and reference
prune have finished. Dialogue collection and review are still running. Once
the reference bank settles, refresh both rebuilt seed IRs, verify, and render
fresh continuation SFT bundles.

The source-generation stage has begun ahead of final dialogue review. Fresh
seed-73 and seed-74 IRs are at
`data/external_pilot/synthetic-s73-pruned-20260921.ir.jsonl` and
`data/external_pilot/synthetic-s74-pruned-20260921.ir.jsonl`. Each has 10,000
schema-audited programs and a matching, stable generator source revision.
Seed 73 has 125 provisional programs and seed 74 has 138. All 144 and 113
occurrences, respectively, of the 80 pruned sale keys are provisional rather
than concrete gold. These are base IRs; after teacher review, refresh their
leaf references, verify the runtime replay, and render training data.
