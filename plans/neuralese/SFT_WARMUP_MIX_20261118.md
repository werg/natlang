# SFT warm-up mix (2026-11-18)

**Goal (owner):** a cheap pure-SFT warm-up from the base 5ff07638 that teaches the tool shape and the kinds of actions
and responses we want. Whatever reasoning precedes a teacher's action is masked. Reasoning is trained afterwards, by
the group-relative term (reward = likelihood of the teacher's action given the student's reasoning, being built in
`grpo.py`) together with GRPO. So the warm-up must leave the student's own reasoning where it is.

**Why a new build:** sft-v3 (run1 + run1-cont from base) cut held-v3 from 47.5% to 31.3%. The cause was trained teacher
and October think blocks: per-call reasoning p50 fell from 1,919 to 350 characters, the student took up the teachers'
summary style, and ran-out rose from 16.6% to 29.6%. The Neuralese text/graph sets also carried 38% of that SFT's loss
on about 94 tasks (DECISIONS 2026-11-18).

- **Data set:** `runs/sft-warmup-20261118-v1`, registered as `sft-warmup-20261118-v1` in
  `training/neuralese_corpora.json`. It holds `chains.pkl`, `val-own.pkl`, `val-teacher.pkl`, `rows.jsonl.zst`,
  `cases.jsonl` and `build.json`, which has the funnel, the mix, loss concentration and every input hash.
- **Builder:** `scripts/neuralese_data/sft_sources.py build`. The invocation is under "Build" below.
- **Launch:** `/home/werg/data/lfm3-rl-dgx-20261029/ops/sft-v4/launch.sh`. It starts `drive.sh`, which runs pin,
  train, behaviour probe and held-v3 on Pop.

## Facets

### 0. The surface: today's runtime, today's answer protocol

On 2026-11-17 at 15:26 (1cb19e291), a plain assistant reply stopped completing an invocation. An answer now goes
through `return_result`, or through `eval {finish: true}`. The system prompt changed with it:
- before: "reply with the declared string as plain text ... A reply without tools ends your turn and is taken as your
  answer";
- after: "Answer with the return_result tool ... A reply without a tool call ends your turn without answering".

The call notes in each invocation's opening message changed too, and so did the tool list of non-editable calls.

grpo-v4-run2 and every held eval on a current pin run on this surface. Most of the earlier SFT sources were recorded
before it: grpo-v2/v3 own successes, teacher pool-v1, October, the sft-v3 sets. Their rows show a retired system prompt
and, in some episodes, a final plain-text answer that is no longer an answer.

**What the build does.** Every episode is replayed through a runtime that has the change (43989d8ba; pool-v2-rich and
part of pool-v1 at ae1f7dacb), and the row is rendered from that replay. A row is therefore today's system prompt,
opening, tools and tool outputs, with the recorded decisions.

**The replay matcher now accepts one kind of difference** (`reworded_key`, 884b1693e). A recorded call may match even
if the runtime reworded its invocation's opening message, as long as the signature line is the same and every action,
tool output and later message is identical. Before this, 2,900 of pool-v1's replays at ae1f7dacb failed on the reworded
call notes alone ("missing" on the first request).

**What drops out.** An episode whose recorded final answer was plain text fails the replay: the runtime asks again and
the recording has no reply. So it is left out, which is right for today's protocol.

### 1. What is a target

Each model call is one row. The prompt (system, task, history, tool output) has zero weight.

| reply | loss |
|---|---|
| the student's own sampled reply (own successes, Neuralese graph rows) | every token: reasoning, `</think>`, text, tool calls, end token |
| a teacher, October or live reply with tool calls | from its first `<tool_call>` token to the end. The think block (visible reasoning or empty) and any text before the first tool call are context. |
| a teacher, October or live reply without tool calls (a final answer) | the answer text and the end token. The think block is context. |
| a written or dataset target (operator seeds, operator consumers' exact answers) | the reply. A leading empty think block is context (`exp0_sft.mask_empty_think`). |

**New in this build: text before the first tool call is context** (`think_context(..., lead_in='context')`, default;
`--teacher-lead-in train` opts out). This text is narration outside the think block. From sft-verified-v4 rows:

| producer | calls with text beside a tool call |
|---|---|
| Claude (Haiku) | 28% |
| OpenHands | 54% |
| Nemotron | 3% |
| Qwen campaign | 10% |
| the student itself | 1.7% |

Held episodes after sft-v3 had text beside 12% of their actions, against the base's 2%. Narration is not an action, and
no outcome selects it. The rule is one function, `serve.targets.non_student_context` (892073ab3). Three consumers use it:
- `sft_sources.think_context`, which the build uses;
- `call_row`, which `grpo.py`'s in-loop teacher demonstrations (`TeacherDemos`) use at their next pin;
- the program-tune fit rows (`tune/rows.turn_graph`). Their teacher turns trained Haiku's reasoning summaries and
  narration at full weight. They are now context for every turn not from the student's own episodes.

The student never receives a gradient toward a teacher's reasoning length or style: `</think>` is trained only in its
own replies. The rows reuse the per-call rendering through `TemplateRenderer`, which is the serving engine's own
`render_request`/`target_items` and was verified text-identical against the actor on 1,500 calls (SFT_SOURCES_20261117).

### 2. Teacher actions: which episodes

**Source.** Verified successes of the three teacher-generation pools:
- `pool-v1`: Luna, Haiku, Nemotron and others; 6,757 successes.
- `pool-v2-rich`: Haiku 4,184 and Luna 2,215 successes so far. `tgen-rich-1` is still generating; 78 families.
- `pool-progtune-decisions`: 373 successes.

**Verification.** An episode is admitted only after an exact replay through a recent runtime with the oracle passing
again:
- pool-v2-rich's 10:03 snapshot and 3,736 pool-v1 episodes at ae1f7dacb, imported from `sft-teacher-demos-20261118`;
- the rest at 43989d8ba, with reworded openings accepted (facet 0).

Both runtimes have today's answer protocol.

**Backend detours are context.** teacher_gen serves teachers through text endpoints. When a teacher's producer
returns a `Neuralese<T>` value, the runtime refuses it (`neuralese-unsupported-backend: ... needs a configured Neuralese
write port`), and the teacher works around the refusal: it retries in eval or returns plain text. The student's actor
serves Neuralese, so the producer's return, which is the action before the refusal, is the student's surface, and the
retries after it are a detour.

So every call whose prompt holds the refusal has zero weight, and the calls before it train (`--backend-workarounds
context`, default). Dropping these episodes instead removed 92% of the composed-Neuralese and 70% of the bgkit teacher
successes in the first build. `--backend-workarounds exclude` or `train` opts out.

**Caps.** Near-duplicates (MinHash 0.8 on actions) are removed. Then:
- per-task caps by base pass rate (at most 4, or 8 for tasks with held constructs);
- producers take turns within a task;
- at most 5% per family (the rich pool's top 10 families hold 67% of its successes);
- dataset wrappers at most 15%.

**Order and budget.** Episodes are taken breadth-first (every task once before any task twice). Within a round, the
families take turns, so a budget that ends mid-round trims the large families. Within a family's turn, tasks with held
constructs come first, then tasks with low base pass rates. Selection stops at the forward budget
(`--teacher-forward-budget`). That gives the most distinct tasks per forward token: the warm-up is about the shape and
kinds of actions, and one more task teaches more than a second solution of a task already seen.

**These rows are also the next stage's data.** "Teacher action given the student's reasoning" is the likelihood the
group-relative term scores, so the warm-up teaches the action distribution that the next stage's reward measures.

### 3. October rows: not used

Every October row predates the answer protocol of 1cb19e291 (facet 0), so none of them is used. Two problems follow:
- Their final replies are plain text (Qwen s72 7% of calls, Luna 0.2-3%), and that is no longer an answer.
- Their system prompt is the retired one.

They also cannot be replayed under today's runtime: they have no task program, and validity there meant only that the
actions compile.

The acquired groups would stay out on their own grounds:
- **CooperBench** uses `exec_command`.
- **OpenResearcher** uses `browser.*`.
- **SWE-rebench OpenHands** uses `bash/edit/read/write`. OpenHands narrates 54% of its calls, and 35% of its calls are
  text-only.

None of these surfaces is the student's, and they are the likely source of the text-beside-action calls seen after
sft-v3.

About 13k verified teacher tasks under today's protocol replace them.

### 4. The student's own reasoning traces (about 25% of the loss)

The advice the owner received was to mix in about 25% high-quality student reasoning traces from a range of tasks.
Here the 25% is a share of the weighted target tokens, because the loss is token-normalized: a token's gradient weight
does not depend on which sequence it is in. As a share of sequences, own traces would take about 80% of the loss,
since an own episode carries about 8k target tokens against a teacher episode's few hundred action tokens.

These traces are on-policy successes, so their loss anchors the reasoning distribution where the base has it. This is
the cheap form of the "small KL to the base" advice: it costs no reference forward, and it is a supervised target the
student already samples.

**Own successes (20%).**
- Runs: grpo-v3-run1 and grpo-v4-run2 only (`--exclude-run grpo-v2/`).
- Why not grpo-v2: run1 and run2 both regressed held-v3 significantly, and grpo-v2-run1's successes reason shorter
  (p50 1,305 characters per call) than grpo-v3-run1's (1,854). The base's held p50 is 1,919.
- Why grpo-v3: grpo-v3-run1 was held +4 pts at steps 7-8.
- Order: breadth-first over tasks, held constructs and low base pass rates first, at most 40k target tokens per task.
- Left out: JOLT+ guided attempts (`--guided exclude`, default; see facet 9) and episodes with imported Neuralese state,
  which cannot be replayed.

**Neuralese graph rows (5%), `sft-neuralese-graph-20261117-v1`.**
- What they are: the student's own successful Neuralese episodes as whole graphs (producers, and consumers with the
  imported state as pass-through documents).
- Families: bgkit reading (pubmed, searchqa, narrative, haystack) and composed Neuralese (relay, typed, fan-in,
  iterate). This is the bgkit-style summarization the owner expects, as student-sampled data.
- Cap: at most 40k target tokens per task. In sft-v3, the same episodes as text rows were 38% of the loss.
- The text set (`sft-neuralese-text-*`) holds the same 449 episodes and is not added on top.
- The imported states cannot be replayed, so these rows keep their recorded rendering, which shows the retired system
  prompt as context. 99% of their replies are tool calls; 37 of 3,384 are text-only.

### 5. Operator rows (3%)

`ops/operators-v1/merged/chains.pkl` holds 5,131 cases of the learned compaction operators: operator seeds and
consumers' exact answers. The share is small because of what the targets are:
- 92% of their target tokens are seeds copied from the source (mechanical extractions, checked spans);
- the consumer targets are dataset answers.

At a large share, copying would become a style. The file on disk still trains `</think>` at target position 0 in all
11,558 rows (it was built before d88e4618f), so the build masks it at load (`think_masked_rows`). Cap: 2,000 target
tokens per task.

### 6. Live workflow rows

`vibestudio-live-corpus/seed-v2`: 26 chains on 2 tasks, observed live, Luna, think blocks already context. All of it is
taken (about 7k tokens).

The v3 seed (5 tasks) lives on Pop only.

### 7. Not used, and why

| source | reason |
|---|---|
| decision rows | Teachers cannot score options, so there are no teacher readout targets. The student's own positive-advantage decisions (1,328 in grpo-v3-run1, on 30 tasks) are logged with a prompt hash only. They would need re-lowering from the rollouts and belong with RL, not a warm-up. |
| hindsight index | 4,776 trajectory summaries: hints for JOLT+, not targets, and drawn from the same teacher pool runs. |
| memetic tasks | Task specs without targets. |
| program-tune fit rows | Their teacher episodes are the progtune-decisions pool, which is in. |
| `sft-neuralese-text-*` | The same episodes as the graph set. |
| October acquired groups | Facet 3. |
| grpo-v4-run1 own successes | Sampled by the regressed sft-v3 start. |

### 8. Held separation, repetition, concentration

- **Held closure:** the held-v3 closure (held-v2 included) is applied to every source:
  - for episodes: ids, split groups, source groups and source ids (`held_closure`, `held_eval.exclusion_ids`);
  - for the aux parts: their own builders' closures, plus a task-id check at load.
- **Repetition:** per-task caps, family shares and the wrapper share keep any one generator from filling the set.
- **Concentration:** `build.json` `mix.loss_concentration` gives the top 10 tasks' share and the top 1%'s share of the
  loss (numbers below).

### 9. Hinted self-distillation: what exists

The owner's form: give the student a good hint about where the solution goes, let it solve the task, then train the
success without the hint.

What exists is JOLT+ teacher attempts (`grpo.py --teacher-attempts`, `rl/jolt.py`):
- The policy itself writes guidance in the same episode, through `applications/guidance-writer`.
  - `writeHint` writes from a verified run in the hindsight index.
  - `writeGuidance` writes from the planner's facts, including the reference solution.
- The guidance is imported as a Neuralese state in a training-only system turn.
- `student_form` removes it from the prompt, and the consumer calls train with a ratio against the guided draw.

Measured on the 11 guided successes that exist (all from grpo-v4-run1; v2/v3 had none, and run2 none yet), it is not
yet the owner's form:
- 4 of 11 replies quote the guidance: 5-word sequences that appear in the guidance but not in the visible prompt.
- 2 of 11 name it ("The user says: Guidance for this call ...").
- In 6 of 11 the guidance already contained the answer:
  - `writeGuidance.nl` asks the writer to "say what a correct result of consumer is";
  - the `reference_solution` fact includes the reference's final `return_result`.
- In student form these replies refer to text that is no longer there, or return an answer with no visible derivation.
- Nothing filters for this, and `checkHint` covers only `writeHint`.
- `teacher_rho` discounts almost no token: median log rho 0, at most 0.37% of tokens below rho 0.1.

So the warm-up excludes guided attempts (`--guided exclude`). In GRPO they remain as they are: that change is a JOLT+
design question for the supervisor and is recorded in DECISIONS.

The owner's form needs three things:
1. hints restricted to *where* the solution goes, never the answer (an exact checker on every guidance path, as
   `checkHint` is for `writeHint`);
2. a filter on trained replies for quotes of the hint;
3. training only on successes whose reasoning stands without the hint.

Details: the hinted-SD audit in DECISIONS 2026-11-18.

## Recipe

| item | value | why |
|---|---|---|
| start | base 5ff07638 (`fresh-init/checkpoint.pt`) | No SFT lineage is a start policy (DECISIONS). |
| trainer | `exp0_sft.py train --chains`, FP32 masters, BF16 compute, `lfm3_fast_train`, shared prefixes, activation 16 GB | The verified SFT path (sft-v3 ran it at 2.8-4.3k computed tok/s, CUDA peak 50-54 GB). |
| epochs | 1 | A warm-up; every episode is distinct. |
| batch | 32k weighted target tokens per step: about 180 sequences and 1.05M chain tokens, about 89 steps | Teacher actions are short against long contexts, so a step at sft-v3's loss-token size already holds 2-4x its sequences. That is the aggressive batch in sequences. A larger loss-token batch would leave fewer than 45 steps. `exp0_sft` now cuts steps on weighted tokens (`chain_mass`); before, masked think tokens filled a third of each batch. |
| LR | 5e-6 peak, 6 warm-up steps, cosine to 10% | sft-v3's rate, whose numerics were sound (its damage was the data). With about 89 steps the total movement is about a quarter of sft-v3's. No sweep fits beside grpo-v4-run2; `LR=` in `launch.sh` overrides. |
| clip | 1.0 | as sft-v3 |
| KL to base | none | The own and graph traces (25%) are the anchor (facet 4). `exp0_sft` has no reference pass, which would add about 40% compute. The probe and val-own below show whether an anchor is needed. |
| validation | `val-own.pkl` and `val-teacher.pkl`: 40 left-over own and teacher successes each, on tasks the build does not train, every 10 steps | val-own NLL rising means the student's own reasoning is moving; val-teacher NLL falling shows action learning that transfers to unseen tasks. |
| after training | behaviour probe v1 (48 base-sampled calls): sampled logp change, entropy, `</think>` hazard; then held-v3 on Pop at 2 samples against the base's Pop measurement (grpo-v3-run1 step 0, as grpo-v4-run1/run2) | sft-v3's probe showed the hazard at 1.02/1k vs the base's 0.54; held is the judge |
| memory | ledger unit `sft-v4-run1`, 72 GB, high priority | It does not fit beside grpo-v4-run2's 99 GB claim. The owner decides when. |

**Success and failure (declared before the run):**
- **Probe:** `behaviour_probe.shifted` bounds: |logp change| ≤ 0.02, |entropy change| ≤ 0.03, `</think>` hazard ratio
  0.67-1.5. Outside them, the checkpoint does not start GRPO until diagnosed.
- **Held-v3 (2 samples, paired with the base):**
  - success: per-task pass rate ≥ base - 2 pts with ran-out not above the base's;
  - failure: below base - 4 pts with p < 0.05, which starts a diagnosis (degradation means bug).
- The warm-up is judged mainly by what it enables: the reasoning-reward stage and GRPO start from it. A warm-up that is
  held-neutral and moves the action format is a success.

## Build

The run directory is `runs/sft-warmup-20261118-v1`, built at 378a9676d.

```
W=/home/werg/data/sft-sources-20261118      # new work dir: episodes (grpo-v3/run1, grpo-v4/run2, 3 teacher pools), replays
sft_sources.py replay --work $W --workers 10 --runtime-repo .../sft-sources-20261117/pin-43989d8baf8a
                                            # after importing the exact ae1f7dacb replays of sft-teacher-demos-20261118
$W/build-v1.sh                              # sft_sources.py build --teacher-forward-budget 90000000 --october-share 0
                                            #   --own-loss-share 0.2 --max-task-target-tokens 40000 --val-episodes 40
                                            #   --aux neuralese-graph=...,share=0.05,task_tokens=40000
                                            #   --aux operators=...,share=0.03,task_tokens=2000  --aux live=...,share=all
sync_training_corpora.py publish --machine dgx --id sft-warmup-20261118-v1
```

**Funnel.**

| | teacher | own |
|---|---|---|
| successes extracted | 13,702 | 2,630 |
| exact replay with oracle success | 11,648 | 2,013 |
| replay failed / inexact (plain-text endings, decision tool loops, drift) | 1,431 / 623 | 127 / 46 |
| imported Neuralese state (not replayable) | | 444 |
| near-duplicates | 341 | 142 |
| wrapper share / family cap / task cap | 1,666 / 60 / 1 | 10 / - / 1,237 |
| over the forward budget / over the loss share | 5,619 | 569 |
| admitted | 3,961 (252 families) | 55 (grpo-v3-run1 50, grpo-v4-run2 5) |

Calls after a Neuralese refusal at zero weight: 2,558.

**Mix (weighted target tokens, `build.json` `mix`).**

| part | chains | weighted target tokens | loss share | forward tokens | forward share |
|---|---|---|---|---|---|
| teacher actions | 15,577 | 2,055,328 | 72.4% | 89.6M | 94.0% |
| own successes (reasoning included) | 92 | 552,418 | 19.4% | 1.17M | 1.2% |
| Neuralese graph (own, bgkit/composed) | 41 | 140,477 | 4.9% | 0.49M | 0.5% |
| operator rows | 535 | 85,919 | 3.0% | 3.80M | 4.0% |
| live workflow rows | 26 | 6,790 | 0.2% | 0.26M | 0.3% |
| **total** | **16,271** | **2,840,932** | | **95.3M** | |

**Teacher producers.**

| producer | episodes | loss share | calls with narration (all now context) |
|---|---|---|---|
| Haiku | 1,662 | 34.4% | 3,218 |
| Luna | 1,921 | 33.3% | |
| Nemotron | 335 | 8.8% | |
| Gemini, GLM | 43 | 1.8% | |

Haiku keeps about half of its target tokens as loss; the rest is think block and narration.

**Episode endings.**
- Teacher: `return_result` 1,936, `eval {finish}` 1,967, text 58 (decision JSON choices).
- Own: `return_result` 53.

**Concentration.**
- 4,250 tasks.
- Top 10 tasks: 8.0% of the loss.
- Top 1% of tasks: 21.3%. These are mostly own episodes: about 10k target tokens each against a teacher episode's
  about 500.
- Largest task: 35k tokens, 1.2%.
- Family cap: at most 29 teacher episodes per family. bgkit 145, composed Neuralese 193, decisions 203.

**Domains (teacher):** relational 1,561, other 1,236, logic 939, actor 214, writing 11.

**Own reasoning:** p25/p50/p75 per call 428 / 1,735 / 4,419 characters. The base's held p50 is 1,919.

**Validation:** `val-own.pkl` (40 episodes, 49 chains) and `val-teacher.pkl` (40 episodes, 189 chains), on tasks the
set does not train.

**Size and time:**
- 95.3M chain tokens, 2.84M weighted target tokens, about 89 steps at 32k.
- About 3 h at sft-v3-cont's measured 8.7k chain tokens/s (prefix sharing). The 4.7 h in `build.json` is the
  5.5k/s estimate without sharing.

**Preflight:** `launch-sft.sh PREFLIGHT_ONLY` against this file checked the sha, gave held overlap 0, and found every
import with its producer document.

## Not checked

- **No training step was run on this file.** The DGX had no 50 GB window beside grpo-v4-run2, and AGENTS says no
  pre-launch smoke gates a run. The format is the one sft-v3 trained, and the preflight passed.
- **LR and batch were not swept.** The recipe row gives the reasoning.
- **No KL to the base.** The own traces and the probe stand in for it.
- **The rich pool keeps growing.** The build used the 11:00 extraction (`manifest_successes_not_extracted` 1,021).
- **Neuralese graph rows are not on today's prompt.** They keep the retired system prompt as context.
- **The `reworded_key` match** accepts any rewording of an invocation's opening beyond its signature line. The task's
  own instructions sit in that message too; an instruction change would pass unseen. Tasks are matched by id, so
  this only matters if a task's text changed under the same id.
- **Writing and chat are thin** (11 teacher episodes). The teacher pools have not covered the new writing families.
- **No `held_eval verify` against exposure.** The task-id/closure checks of the build and the preflight are what
  separates held data.
- **October rows.** No re-render to today's protocol was tried: no task program, so no replay.
- **JOLT+ hint leakage in GRPO** is not fixed here (facet 9).
- **Program-tune fits** with the new non-student masking were not re-run.
- **Error actions.** Teacher actions that raised a runtime error are not masked. Their rate is 2-5% of calls, and no
  structured error flag exists to key on.
