# The harness bench: a Neuralese student with harness support, trained from strong agents' trajectories

Owner direction, 2026-10-09: the pi port (applications/pi) with its companion (applications/pi/COMPANION.md) is a
training bench for Neuralese. Strong agents' trajectories are enriched with what a helpful harness adds, and the small
Neuralese students learn to work like the teacher with that support. The harness adds two kinds of input to the
executing model's context:

1. **Background help**: what the companion finds by researching beside the agent, delivered as Neuralese where it
   would have arrived live.
2. **Pre-digested tool results**: a large tool output reaches the agent as a short Neuralese digest, written by the
   harness from the full output and conditioned on what the agent wanted from the call; the full output stays
   recallable.

The students are **Mellum** (Mellum2.1-12B-A2.5B-Thinking, as a nested family: every member a real objective, see
MAPLE_NESTED.md and DECISIONS 2026-10-09) and **LFM2.5-350M**. Both use the same recipe, records and code; text is
rendered per tokenizer, and only backbone-inherent differences are declared. All training follows TRAINING_RECIPE.md.
Data enters through the corpus registry with manifests; admission is explicit, per record, by stated criteria.

## 1. What the data needs, and why most of it is static

Measured on 300 SWE-rebench OpenHands trajectories (median 185k characters each): tool output is 85% of a
trajectory's text. A quarter of the tool outputs (those over 2,000 characters) hold 79% of it; outputs over 6,000
characters are 9% of the outputs and 58% of the text. Digests of large outputs therefore address most of the context,
and they are the part that needs nothing but the recorded trajectory.

| Input | Source | Static? |
| --- | --- | --- |
| Agent turns (reasoning, text, tool calls) | the teacher's trajectory | yes |
| Tool observations, in pi's own format | replay of read/edit/write in pi-durable's tools over the base commit; bash output as recorded | yes (no container, no model) |
| Tool-output digests | the digest operator, trained through the student (§3) | yes: the full output is recorded |
| Background offers | the real companion, run offline on replayed workspace states (§4) | offline model runs |

## 2. Pipeline (implemented)

`training/neuralese/scripts/harness_bench_build.sh OUT [LIMIT] [OFFSET]`, run under the memory ledger:

1. **Surface.** `natlang run applications/pi -- surface --cwd /workspace/project --companion` exports pi's system prompt
   and tool schemas (applications/pi/surface.ts), so records always match the live harness.
2. **Prepare** (`natlang_neuralese.harness_bench.prepare`). Normalizes each OpenHands trajectory into a pi transcript
   (`openhands.py`, dialect `openhands-v0.54`, mapping `pi`). Joins it with its SWE-rebench task (repository, base
   commit). Renames the task's workspace to `/workspace/project` throughout. A rename is all it is, and it lets every
   record share one system prompt piece. Gold and test patches are never copied: they are hindsight-only.
3. **Replay** (`natlang run applications/pi -- replay`, applications/pi/bench/replay.ts). Walks each trajectory over a
   checkout of its base commit:
   - read, edit and write run pi-durable's own tools, so the observations are byte-for-byte what pi shows (through
     `appendToolResult`, diagnostics included), and the checkout follows the teacher's edits;
   - a `view` without a range becomes what the checkout says it is: a read of a file, or a listing of a directory that
     runs in the checkout;
   - bash keeps the teacher's recorded output (only its environment could produce it), reformatted the way pi formats
     bash: OpenHands' exit-code and cwd trailers become pi's error diagnostic, bounded to the tail.

   Every step is verified. A read must show exactly the text the teacher saw, and an edit or write must succeed or
   fail as the teacher's did. At the first disagreement (a command changed files, or tool semantics differ), replay
   stops and reports the step; earlier steps are verified. Repositories are bare partial clones cached under
   `~/data/harness-bench/repos` (NVMe).
4. **Records** (`natlang_neuralese.harness_bench.records`). Writes `natlang.teacher_training_turn.native/1`, the
   pipeline's own format, which both students' text warm-up and recurrence stages read:
   - one record per supervised assistant turn: the system prompt as soft piece `prompt:pi-agent#sha256:…`, the task,
     the earlier turns (reasoning in `reasoning_content`), the turn as target, and pi's tools;
   - a tool result over 2,000 characters becomes a **digest part** (DECISIONS 43), as in §3;
   - only steps before a replay divergence are kept;
   - splits are by repository (5% test), and source groups are the repository and the instance;
   - admission: resolved task, pi-replayed and verified, and exact mapping (`harness-bench-criteria/2`); anything else
     is held with its reason;
   - `pieces.jsonl` holds the system prompt and `prompt:digest`;
   - by default 8 targets per trajectory: first the turns that read a digested output, then others; `--targets all`
     gives whole-trajectory supervision.

## 3. Tool-output digests: the digest operator, conditioned on the agent's intent

A digest part in a record:

```json
{"type": "digest", "name": "digest:<sha12>", "holder": "recall(\"<call id>\")", "value_type": "string",
 "source": "<the full output>", "preview": "<the output as the teacher saw it (cut at 40,000 characters)>",
 "instructions": "The agent made this tool call and reads its output next: <call> Its reasoning when it made the call: …",
 "note": "  // digest of the output; recall(\"<call id>\") returns all of it"}
```

`instructions` and `note` are per-part (trainer: `train/trajectories.py`; TypeScript: `compiler/neuralese-conversion.ts`).
The digest is conditioned on what the agent knew and wanted at the call, never on later turns. That is the "Neuralese
input" of the digesting function. Once the student's raw port is runtime-qualified for its exact weights, the
intent's own text can be encoded through the port as a block.

Training uses the existing trajectory trainer with no new objective:

- **Text renderings** (text warm-up, `--digest preview`) show `preview`: the student learns from the teacher's own
  view.
- **`--digest written`** writes the digest through the port's differentiable write procedure at every step, from
  `source` at the operator's write site, conditioned on `instructions`, and chunked when long. The reader's loss on the
  teacher's following actions trains the writer, so the digest keeps what those actions needed.
- **`--distill`** adds self-distillation from the same model reading `preview`, which for most outputs is the full
  output. The digest is thereby taught to act like the full output, at a fraction of its context. No summaries are
  fabricated: relevance is defined by what the trajectory did next.
- **Size**: `--tokens-per-vector` sizes a write from the preview it replaces, and the stop head learns the boundary;
  at inference the stop head decides.

## 4. Background offers

The companion's research runs offline on replayed workspace states. Replay gives exact files at each verified step,
and the companion's file, search and list services gain a snapshot-backed implementation. It sees only the
transcript prefix, the workspace at that step, and allowed documentation. Its executor can be a strong model, since
offline research has no latency limit.

- **Hindsight** scores and selects offers under a context budget: did an offer contain what the teacher later spent
  steps finding? Hindsight never writes an offer's content.
- **Leakage audit**: an offer whose content appears only in the future is rejected.
- An **oracle** variant with future information exists only as a labelled upper bound for evaluation.
- **Rewrites**:
  - *augmented*: same actions, offers injected at the briefing section;
  - *shortcut*: exploration an offer makes unnecessary is removed, when the remaining actions still reach the outcome,
    and only for offers whose reproduction rate is measured high;
  - *compressed*: old observations become digests and recall handles.
- The writer of Neuralese offers trains like the digest writer: through the student's loss on the following teacher
  actions.

## 5. Harness changes (each useful on its own)

| Change | Status |
| --- | --- |
| Agent surface export (system prompt and tool schemas) | done (`surface.ts`) |
| Replay in pi's tools, with verification | done (`bench/replay.ts`) |
| Companion output shaping as a pluggable hot path (crisp head and tail, or `shape.nl`), with `recall` | done |
| Per-part digest instructions and note in the trainer and the conversion format | done |
| A Neuralese content part in pi messages: a block reference plus a text gist, read by a Neuralese-capable provider and shown as the gist elsewhere | next |
| Tool-output digest as a third shaping mode: the runtime's `neuralese.digest` with the call's intent as instructions | next (after the student serves) |
| A purpose on tool calls (the agent's stated intent), which the digest and shaping condition on | next |
| A pi-ai provider for the Neuralese server (the student serving pi) | next |
| One record format live and offline: live companion runs write the same records through the call store | next |
| Snapshot-backed companion services for offline research | next (with §4) |

## 6. Integration into both students' training

Harness records are ordinary native records, so they enter the existing stages:

1. **Text warm-up** (`core_text_warmup`, `ar_feedback_fixup`). The shared renderer renders the records per tokenizer
   (LFM and Mellum twins) with digests as previews, giving whole-trajectory supervision with tool output at lower
   weight (owner decision). Render only with the fixed renderer (Pop's 2026-10-09 history-boundary correction).
2. **Recurrence / trajectory training** (`raw_recurrence_training`, `train.trajectories`) with `--digest written
   --distill`. This consumes the exact runtime-qualified weights of each student, as the recipe requires. For Mellum,
   every nested member is an objective.
3. **Evaluation** live in pi on held-out repositories, with the student behind the Neuralese provider:

   | Configuration | Purpose |
   | --- | --- |
   | teacher alone | reference |
   | student alone | the gap |
   | student + text companion | how much harness support helps |
   | student + Neuralese companion and digests | the same at a lower context cost |
   | student + oracle companion (labelled) | upper bound for the companion's information |

   Report task success, steps, context tokens, wall time and offer usefulness.

Recipe entries for both students are declared once the slice is reviewed: a `harness_bench` cohort in the recipe's
cohort map, mixed into the existing native cohorts, never replacing them.

## 7. Order of work

1. Slice: 20, then 200 trajectories through build.sh; review the replay divergence reasons and the records; register as
   a derived corpus with a manifest.
2. Render text twins for LFM and Mellum; declare the cohort in both recipes' text warm-up.
3. Digest training in the recurrence stage, once each student's raw port is qualified.
4. Offline companion offers (§4) on replayed states; text offers first, then Neuralese.
5. The live harness changes in §5, then evaluation (§6).

**First build (step 1 done), 2026-10-09/10:** `build-20261009-1000` at code commit 744bdd4f, surface sha256 c8e3d62f…,
registered as `harness-bench-swe-rebench-openhands-pi-records-20261010-v1` (manifest in `training/corpus-manifests/`,
files under `data/neuralese/corpora/`). Corpus-level training admission is held pending text-twin rendering with the
fixed renderer and the recipe cohort declaration (step 2); per-record admission is in the records.

| Measure | Count |
| --- | --- |
| Resolved trajectories prepared | 492 |
| Replayed | 489 (3 clone failures, NREL/hescore-hpxml) |
| Fully verified | 305 |
| Verified up to the first divergence | 184 |
| Steps verified | 22,538 of 22,722 attempted |
| Records (all pass `harness-bench-criteria/2`) | 3,852 (3,740 train / 112 test) |
| Digest parts | 20,359 |

Divergences (first per trajectory): `task_tracker` 113, `execute_bash` interactive input 31, read shows other text 22,
read failed in pi 8, write failed for the teacher 5, read failed for the teacher 4, other `str_replace_editor` 1.
Tool calls lost after a divergence: 1,487 of 6,089 in the 113 `task_tracker` trajectories, 1,151 of 1,895 in the 31
interactive-input ones (§8).

## 8. Open questions

- Replay divergence from bash commands that change files (scripts, `sed -i`, `git apply`). Running such commands
  requires the task's environment image (x86_64; Pop, or emulation). Measure the divergence rate first.
- **OpenHands `task_tracker` (113 trajectories cut, 1,487 tool calls lost).** The teacher calls it 335 times in the
  492 prepared trajectories: `plan` (179) writes a list of `{id, title, status: todo|in_progress|done}`, and `view`
  (156) shows it back. pi has no equivalent: the surface is read, write, edit, bash and the companion's recall;
  `extensions/` has coding-tools, companion and subagent; pi-durable ships a todo document only as a test example
  (examples/11-extension-state.ts) and a plan mode only as example 27. So nothing maps exactly, and nothing was
  implemented. The options, for the owner:
  1. *Plan as thinking* (the precedent is `think`, which already becomes a thinking block): a `plan` call becomes a
     thinking block that lists the items with their status; a `view` call and its result are dropped (the plan is
     still in the earlier reasoning). pi's surface does not change and the student learns to plan in its reasoning,
     which is what it can do in pi. The cost: the action is rewritten, so these steps fail `exact-mapping`; they need
     their own criterion (e.g. `rewritten:task_tracker-as-thinking`) and an admission decision. A turn whose only
     call was `view` disappears, so the teacher's next turn follows a result it never saw. Gain: up to 1,487 calls
     and up to 113 fully verified trajectories (fewer where a later step diverges).
  2. *Plan as a file* (`write` of a plan file outside the repository): a pi idiom, but a fabricated action whose
     file the teacher's later recorded outputs (`git status`, `ls`) do not show.
  3. *Plan in the companion*: the companion's session state already holds "goal, plan, hypotheses" (COMPANION.md);
     the teacher's plan becomes the companion briefing's plan from that step on, and the call is dropped. This fits
     the bench's purpose (the harness carries the state), but it is an offline-companion augmentation (§4), not a
     mapping, and it teaches the agent to rely on a briefing pi gives only with the companion on.
  4. *A todo tool in pi* (a real extension from pi-durable's example 11, or the dialect tool registered in the
     records' surface): exact, but it changes pi's agent surface, so it is the owner's decision. A dialect tool that
     pi does not have trains the student on a surface it will not run in.
  5. *Truncate* (today): exact, but it removes planning trajectories, which are probably the longer, harder tasks.
  Proposal: 1 for training data now, behind its own criterion; 4 only if the owner wants a todo tool in pi.
- **`execute_bash` with `is_input` (31 trajectories cut, 1,151 tool calls lost).** Interactive input to a command that
  is still running: `C-c` 52 times, `q` 2, a command line 1 in the prepared set. It follows OpenHands' soft
  timeout ("no new output after 30 seconds … send keys …"). The commands that hang are pagers (`help(...)` under
  `python -c`), REPL sessions the teacher typed into, long test runs, installs and servers. pi's bash runs with stdin
  ignored and no terminal, has no default timeout, and its only interruption is the `timeout` argument ("Command timed
  out after N seconds") or an abort ("Command aborted"). There is no exact mapping:
  - a pager or `input()` does not hang in pi at all (no tty: the help text prints, `input()` gets EOF), so the true pi
    observation needs the command re-run in the task's environment (the bash open question above);
  - rewriting the hung call to carry `timeout: 30` with pi's timeout result and dropping the `C-c` turn reads well,
    but adds an argument the teacher did not choose, and it is wrong for pagers (pi would have returned output);
  - a `bash_input` / process-session tool (like Codex's write_stdin) changes pi's surface: owner's decision.
  Proposal: keep truncating, but one step earlier. Today the hung call's step is kept, and its observation carries
  OpenHands' soft-timeout instructions ("send keys (C-c, C-z, C-d) … use the timeout parameter in execute_bash"),
  which describe a tool pi does not have; 6 records of this build show that text. Replay should treat a soft-timeout
  observation as the divergence (reason `bash interactive (soft timeout)`), and environment re-execution later
  recovers these trajectories with pi's own observations.
- Cross-tokenizer teachers: supervision is on actions (text) and the students' own self-distillation, never logit KL
  to the teacher.
- Record size: a record carries its whole prefix (median about 160k characters before digesting). Target sampling
  bounds the count. The trainer's `max_tokens` drops records that remain too long in crisp form.
