# SFT sources for the base-checkpoint SFT experiment (2026-11-17)

**Experiment:** one SFT run from the base LFM3 checkpoint (5ff07638) on verified, diverse trajectories. It is judged by
a paired held-v3 eval and starts when grpo-v3-run1 frees the DGX (around 2026-11-18).

**This document:** the training-data inventory for that run: what each source is, which filters it passed, and what
the built set contains.

- Builder: `scripts/neuralese_data/sft_sources.py` (`extract`, `replay`, `check-render`, `build`).
- Built set: `runs/sft-verified-20261117-v1`, registered as `sft-verified-20261117-v1` in
  `training/neuralese_corpora.json`, with manifest `training/corpus-manifests/sft-verified-20261117-v1.json`.
- Working data: `/home/werg/data/sft-sources-20261117/`.

## The built set (v1)

**Size:**

| measure | value |
|---|---|
| trajectories | 2,220 |
| tasks | 1,945 |
| `exp0_sft.py train --chains` sequences | 6,226 |
| forward tokens (one epoch) | 47.7M |
| target tokens | 5.76M |
| epoch time at about 5.5k tokens/s on the GB10 fast path | about 2.4 h |

**Files:**
- `chains.pkl`: the `exp0_sft --chains` input. Each call's prompt has zero weight. A call's segments carry `weight` only
  where a reasoning-free teacher reply's empty think block is masked.
- `rows.jsonl.zst`: every call's semantic `messages`, `tools` and `target`, plus its source, task, producer, token counts
  and `loss_from`. This is what a later re-render starts from.
- `cases.jsonl`: the task IR (`natlang.program/2`) of the GRPO-pool tasks in the set.
- `repetition.json`: `measure_repetition.py` over those tasks.
- `build.json`: the per-filter funnel, the per-source and per-producer statistics, the October budget passes, and the
  input hashes.

**Shares of the set:**

| source | trajectories | calls | weighted target tokens | share of weighted target | forward tokens | share of forward |
|---|---|---|---|---|---|---|
| own GRPO successes | 520 | 4,369 | 4,160,027 | 72.3% | 9.64M | 20.2% |
| teacher successes (GRPO pool) | 998 | 5,589 | 832,457 | 14.5% | 19.23M | 40.3% |
| October teacher / acquired rows | 702 | 4,744 | 760,530 | 13.2% | 18.86M | 39.5% |

Own successes dominate the target tokens because the student reasons at length: about 8k target tokens per episode,
against about 830 for a teacher episode.

**Per producer** (weighted target-token share; reasoning-free calls whose think block is context, not loss):

| producer | trajectories | weighted target tokens | share | think-context calls |
|---|---|---|---|---|
| own grpo-v3 run1 | 232 | 2,108,216 | 36.7% | 0 |
| own grpo-v2 run1 | 241 | 1,565,763 | 27.2% | 0 |
| own grpo-v2 run2 | 47 | 486,048 | 8.5% | 0 |
| nemotron-3-ultra (paid) | 184 | 303,402 | 5.3% | 0 |
| gpt-6-luna (teacher pool) | 671 | 278,680 | 4.8% | 3,521 |
| nemotron-3-ultra (free) | 126 | 207,999 | 3.6% | 0 |
| October acq OpenResearcher | 130 | 140,857 | 2.4% | 0 |
| October Qwen3.6-35B (s72, campaign, fallow) | 149 | 201,614 | 3.5% | 84 |
| October gpt-6-luna (curricula, cohorts, children, ep, fallow, projection) | 259 | 231,303 | 4.0% | 2 |
| October acq SWE-rebench OpenHands (Qwen3-Coder-480B) | 103 | 83,324 | 1.4% | 230 |
| October acq CooperBench (Codex) | 33 | 80,944 | 1.4% | 416 |
| glm-4.7-flash, gemini-3.5/3.7-flash | 17 | 42,376 | 0.7% | 64 |
| October Bonsai-27B and stealth teachers (fallow, projection) | 28 | 22,488 | 0.4% | 29 |

**Coverage:**

| source | families | domains (task curriculum) |
|---|---|---|
| own GRPO successes | 146 | relational 331, other 103, logic 50, actor 36 |
| teacher successes | 229 | logic 343, relational 335, other 273, actor 43, writing 4 |
| October | 110 across 12 groups | |

- The 1,243 GRPO-pool tasks in the set span 84 families: 202.6 effective instruction texts, 205.6 effective programs and
  15.2 effective declared structures.
- Generators: 48% external datasets wrapped as typed root functions, 21% inline curriculum, 9.5% authored tasks, the
  rest codebase, composed, bgkit and decision tasks (`repetition.json`).
- The dataset-wrapper task framing ("Solve puzzle. Use eval ...") recurs. Every action and answer in these trajectories
  is the model's own, and is not scripted.

**Too-long calls:** 476 calls (99 of them own) exceed `exp0_sft.MAX_LEN` (32,000 tokens). `chains_of` leaves them out,
and the rest of their episode still trains.

## Sources and filters

### 1. Own GRPO successes

The successes of grpo-v2 run1, grpo-v2 run2 and grpo-v3 run1.

- **Where they live:** `ops/grpo-v*/run*/groups/*` (cases, rewards) plus the rollout archives
  `/mnt/external/natlang-grpo-v*/run*/rollouts/g*.rollouts.jsonl.zst`. A retired group's rollouts exist only in the
  archive.
- **Record format:** `natlang.rollout/1`. Each model call carries its semantic `messages`, `tools` and `target`, plus the
  exact actor prompt and sampled ids. Every group's `cases.jsonl` keeps the task IR, and each episode keeps its seed.

| filter | grpo-v2 run1 | grpo-v2 run2 | grpo-v3 run1 | total |
|---|---|---|---|---|
| scored episodes | 2,766 | 1,158 | 3,420 | 7,344 |
| successes (quality 1) | 1,393 | 573 | 1,871 | 3,837 |
| Neuralese native-state calls (excluded) | | | | 449 |
| replay failed (decision readouts, others) | | | | 168 |
| replay succeeded but not exactly | | | | 255 |
| exact replay, success | | | | 2,965 |
| held closure | | | | 0 |
| near-duplicate | | | | 0 |
| over the per-task cap | | | | 2,445 |
| admitted | 241 | 47 | 232 | 520 |

The near-duplicate test is MinHash at Jaccard 0.8 on the actions. The per-task cap concentrates these 520 episodes on
the tasks the base rarely solves. The mean base pass rate of all admitted episodes with a measurement is 0.49.

### 2. Teacher generation on the GRPO pool

- **Where it lives:** `ops/teacher-gen-20261115/pool-v1`: `manifest.jsonl` and `episodes/t*/` (`rollouts.jsonl` and
  `cases.jsonl`).
- **Record format:** `natlang.rollout/1` without sampling ids. The pool is still growing.
- **Snapshot of this build:** 1,199 successes, re-extracted at 02:40. The manifest held 1,517 successes when the build
  ran (318 more arrived during the replay).

| filter | count |
|---|---|
| successes extracted | 1,199 |
| replay failed | 123 |
| replay inexact | 59 |
| exact replay, success | 1,017 |
| near-duplicate | 19 |
| admitted | 998 |

Teacher replies arrive at the provider's sampling settings. Luna returns no reasoning: 3,521 Luna calls (and 64 Gemini
calls) render the template's empty think block, which is masked as zero-weight context (see "Loss").

### 3. October corpora

Classified by a subagent; summaries in `/home/werg/data/sft-sources-20261117/october/summary*.json`.

**Admitted groups.** These have semantic messages, tools and target, and their actions are real teacher decisions.

- The natlang groups were rebuilt for today's tool surface with `serving_prompt.serving_messages`. The rebuild drops
  rows that call tools the surface no longer offers.
- About 10.2k rows with the old surface were set aside (`*.stale-surface.jsonl.zst`).
- Each eval action is compiled with the realm's own compiler (`ts-host/scripts/check-eval-actions.mjs`). Of 128,371
  eval actions in 226,856 natlang rows, one fails (luna-curricula).

| group | candidate rows | candidate tasks | trajectories admitted | calls | forward tokens |
|---|---|---|---|---|---|
| luna-curricula (gpt-6-luna) | 64,442 | 9,516 | 132 | 897 | 3.18M |
| qwen-s72 (Qwen3.6-35B) | 69,921 | 2,138 | 40 | 836 | 2.67M |
| fallow-v13r2 (teacher rows only) | 68,288 | 1,289 | 58 | 405 | 2.43M |
| acq-d32-openresearcher | 30,412 | 6,093 | 130 | 646 | 2.19M |
| acq-swe-rebench-openhands (resolved, replay-verified upstream) | 64,651 | 2,910 | 103 | 264 | 1.96M |
| luna-early-cohorts | 1,708 | 191 | 32 | 277 | 1.19M |
| luna-v19-decision-children | 281 | 49 | 37 | 200 | 1.12M |
| qwen-campaign | 20,475 | 8,111 | 90 | 326 | 1.09M |
| student-projection (teacher rows only; 242 non-teacher rows dropped) | 1,484 | 142 | 14 | 207 | 1.00M |
| acq-d13-cooperbench | 23,165 | 830 | 33 | 429 | 0.80M |
| luna-ep-campaigns | 182 | ~11 | 10 | 182 | 0.75M |
| luna-clinc-children | 75 | 23 | 23 | 75 | 0.48M |

How these were selected:
1. Held: the held closure was already removed from the candidates.
2. Near-duplicates: dropped against everything admitted before.
3. Per task: one trajectory.
4. Per family: at most 5% of the October trajectories.
5. Budget: October takes 40% of the forward tokens, split by group weight (`DEFAULT_OCTOBER_GROUPS`; Luna 3, Qwen-s72,
   fallow, SWE and OpenResearcher 2, the rest 1).
6. Size estimates: each pass renders its picks and measures their actual forward tokens. Prefix-extending calls share a
   chain, so the actual tokens run at 0.07-0.70 of the character estimate. Two passes reached 18.86M of the 19.25M
   budget.

**Excluded October sources:**

| source | reason |
|---|---|
| external-expansion-* (search-owl, evidence-terminal, codeagent, first-wave, agentic-coding, rightfit, gatc, deepdive) | Scripted wrapper (`natlang.port_record_calls/1`): constant "Do what item.request asks…" instruction, target `eval {"code":"item.evidence"}`, model `port-record-reference`, 0 provider calls. |
| port-record-calls-s3 | Same scripted wrapper. |
| FALLOW 22 fetaqa/2wiki/bird chains | Same scripted wrapper. |
| broad-graphs | Same scripted wrapper. |
| big2 port-record communication graphs | Scripted replay (RECIPE_REVIEW_20261103: about 37% of the old mixture's loss). |
| static-lambda expansion/rationalized, neuralese-e2e-cohort, bgkit-knowledge-rationalized, clef-pools, clef-new, static-s7k, pop-static folder-edit / recovery-loop | Generator-written actions (`static-lambda-reference`, `replay-demonstrations.mjs` "No model calls"). |
| gemini-clinc-static, gemini-semantic-mixed, luna-v43 static controller, luna-v17-v9 portable-context view | Authored static programs around saved labels. |
| refine-judge | Labels from a crisp checker; no model decisions. |
| D12 | The whole transcript sits in one tool block; only the closing summary would be a target. |
| D16, D18 | Targets copied from dataset labels (MAST flags). |
| D15 | Raw command text without a tool protocol. |
| acq-d14-cooperbench-qwen | Qwen-9B, 61% exact duplicates. |
| Luna native-episode graph leaves (sales, coedit, tatqa, designbrief, telecom, relational, deep-codebases v6, t2-*-luna-ordinary), qwen15 boot chains, loghub | Rendered token ids only; a re-conversion from the campaign captures would be needed. |
| *.stale-surface rows (~10.2k, mostly Qwen campaign invoice processing) | Old system prompt and tools (`delegate`, `python`, `transcript.*`) that the current surface cannot rebuild. |

## Validity: replay through the current runtime

`sft_sources.py replay` runs each successful episode's task again under the pinned runtime of main aa1bd846
(`pin_checkout.py` copy). It uses the same path as GRPO and teacher_gen:
1. `task_rollouts.prepare` (student root, full rendering, the recorded seed).
2. `run_bonsai_queue.py` against a local replay server.
3. `task_rollouts.score` (the exact oracle).

The replay server answers each model request with the recorded reply of the call whose visible history matches. It
ignores the system prompt and tool-call ids.

An episode is admitted only when all of these hold:
- the oracle passes again;
- every request was matched exactly, with no drifted or unanswered request;
- the turn count equals the recording.

The admitted rows are the current runtime's record of the same decisions. In every exact replay the system prompt and
tool schemas equal the recorded ones.

Why replays fail or are inexact:

| cause | details |
|---|---|
| Neuralese native-state calls | Their imported states live only in the actor store, and the store collects them after training. 449 own episodes; they cannot be text rows anyway. |
| Decision readouts (`decide`) | Need the actor's scored readout; a text endpoint answers them as a tool loop with another history. |
| Live web tools (search pages) | The pages changed. The drifted turns are recorded per episode in `replay/<name>/replay.json`. |
| Neuralese producer calls | No text reply to serve. |

Teacher replays that fail are mostly early teacher_gen episodes recorded under an older tool surface (system or tools
changed).

**Renderer check** (`check-render`, 1,500 own calls). `TemplateRenderer` is the serving engine's own
`render_request`/`target_items` on the tokenizer alone (the actor's tokenizer and chat template, `preserve_thinking`).

| check | result | notes |
|---|---|---|
| prompts, text-identical to the actor's | 1,500 / 1,500 | |
| prompts, token-identical | 951 / 1,500 | A later call's history holds earlier replies as sampled, which can tokenize differently from the canonical text. |
| replies, text-identical | 1,467 / 1,500 | |
| replies, token-identical | 1,259 / 1,500 | |

The 33 replies that are not text-identical:
- Decision readout calls with no sampled text.
- A few where the template re-serializes a JSON value the model wrote with other spacing.

The trained target is the canonical tokenization.

## Loss

Each model call is one row:
- The prompt (system, task, history, tool output) is zero-weight context.
- The reply is the target: reasoning, text, tool calls and `<|im_end|>`.

For a reply that a teacher or an October source gave without reasoning (Luna, Gemini, some Qwen, Codex and OpenHands
calls), the template still renders an empty think block. `</think>` and the whitespace after it are zero-weight
context, and loss starts at the action (`reasoning_free_context`; test
`test_a_teacher_reply_without_reasoning_trains_only_the_action_after_the_empty_think_block`; checked on real Luna rows,
e.g. context `'</think>'`, target `'<tool_call>\n<function=eval>…'`).

Training the empty block would teach the student to close its reasoning at once, and no verified outcome selects that.
The student's own replies keep every sampled token.

## Neuralese successes as text rows (`sft-neuralese-text-20261117-v1`)

`build` drops the 449 own successes with imported Neuralese state whole. `sft_sources.py neuralese-text --work W --out
OUT --tokenizer W/tokenizer` keeps every call of them whose messages hold no `native_state` part:
- the producer invocations (`f(...): Neuralese<T>`): the actor captures the state over the producer's own sampled value
  span (`sglang_actor._sglang_capture`, ids[start:end] of the producer's sequence), so these rows train exactly the
  tokens whose keys and values the consumer imports;
- the consumer's calls before its first import.

A call after an import needs the imported state as context; it is counted (`calls_after_import`), not written.

| | episodes | tasks | calls | target tokens | forward tokens | calls after import (skipped) |
|---|---|---|---|---|---|---|
| bgkit KR readers | 115 | | 1,642 | 2.03M | 5.22M | 820 (0.86M target) |
| composed Neuralese and other | 148 | | 2,164 | 2.79M | 6.49M | 916 (0.84M target) |
| total | 263 | 94 | 3,806 | 4.82M | 11.7M (1,115 chains) | 1,736 |

Validity: oracle success at collection (grpo-v2 run1/run2, grpo-v3 run1); held-v3/v2 closure (0 hits); each call's
system prompt and tool schemas equal ones the current runtime rendered in an exact replay (0 mismatches); per-task cap 4
(186 over the cap). Not replayed: the imported states lived only in the actor store.

## Held exclusion

- **GRPO-pool tasks:** `held_eval.exclusion_ids(held-v3)` (which covers held-v2) plus `held_closure`: the held tasks'
  ids, split groups, source groups and source ids, for held-v3 and every earlier set it excludes. 0 of the verified
  episodes hit it; GRPO and teacher_gen had already excluded the closure.
- **October rows:** the subagent used `lower_sft_sources.held_probes` and text shingles (2,307 Luna and 1,159 Qwen-s72
  rows dropped, among others), and the build applies the id and group closure again (0 more).

## Re-rendering later: what sources 1 and 2 keep

Every own and teacher episode keeps:
- the task IR (group `cases.jsonl`);
- the seed;
- each call's semantic messages, tools and target (reasoning, text, raw tool calls) and its invocation id;
- the execution graph;
- the oracle verdict.

The replay above shows this is enough to re-run the decisions under a changed runtime and re-check them.

**Gap closed (75cbae630):** a rollout record did not say which runtime produced it. Only the run log did, and
teacher_gen recorded it only in its launch log. `task_rollouts.score` now writes `code` (commit and dirty flag) and
`model_facing` into every `natlang.rollout/1` record. This covers the RL collector and teacher_gen, which share it, and
enters at the next pin.

**Not recoverable by design:**
- Native Neuralese state blocks: the actor store collects them after training, and a new policy would produce them anew.
- Decision readouts: only their option scores are kept, which the actor rescores.

**Materializer:** `ts-host/scripts/materialize-native-teacher.mjs` has not rotted. Its 61 tests pass, and it
materializes the September `natlang.teacher_trajectory.native/1` replays (nice, range, ticks: 92 turn records). It reads
the retired teacher-collector format, though. Today's producers (GRPO, teacher_gen) write `natlang.rollout/1`, whose
turns are already template-neutral, so `sft_sources.py replay` plays the materializer's re-rendering role for them.

## Launch recipe (rebuild with the current teacher pool)

Luna adds about 1,000 teacher episodes an hour (about 70% verified), so the launch build re-extracts the pool first.
The scripts are in `/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/`.
Run them under the ledger at about 3.5 GB: two replay workers, since one sokoban teacher episode replays at 2.8 GB.

```
sft_sources.py extract --work W --teacher ops/teacher-gen-20261115/pool-v1 --out-name episodes-teacher
sft_sources.py replay  --work W --workers 2 --runtime-repo W/pin-<main commit>
sft_sources.py build   --work W --out runs/sft-verified-<date>-vN --tokenizer W/tokenizer --october W/october
sync_training_corpora.py publish --machine dgx --id sft-verified-<date>-vN
```

`build` refuses a manifest whose successes are not all extracted unless `--allow-stale-teacher` is given (v1 used it:
the pool grew during the replay). The task caps let producers take turns, so Luna's repeats of a task never fill it
alone. `build.json` reports each producer's share.

## Verified, not verified

**Verified:**
- Exact replay and oracle success for every own and teacher episode admitted.
- The renderer against the actor's token captures (text identical).
- The think-block masking on real rows.
- October eval actions compile under today's compiler, and October tools exist on today's surface.
- The held closure.
- `measure_repetition.py` on the admitted GRPO-pool tasks.

**Not verified:**
- October actions were not replayed: there is no task program for most of them, and validity means compiled code and
  known tools only.
- For the acquired groups (SWE, OpenResearcher, CooperBench) only the upstream verification exists (resolved,
  replay-verified upstream for SWE); their bash and pi errors were not filtered.
- The 2,307 Luna rows dropped by the held text-shingle check were not reviewed: they may be over-excluded.
- No training was run: the trainer's consumption of `chains.pkl` was checked only through `exp0_sft.chains_of` and
  `chain_weights` in tests.
