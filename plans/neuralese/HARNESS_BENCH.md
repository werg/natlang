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

## 8. Open questions

- Replay divergence from bash commands that change files (scripts, `sed -i`, `git apply`). Running such commands
  requires the task's environment image (x86_64; Pop, or emulation). Measure the divergence rate first.
- Cross-tokenizer teachers: supervision is on actions (text) and the students' own self-distillation, never logit KL
  to the teacher.
- Record size: a record carries its whole prefix (median about 160k characters before digesting). Target sampling
  bounds the count. The trainer's `max_tokens` drops records that remain too long in crisp form.
