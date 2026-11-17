# Learned compaction operators: inventory, staleness and training on today's stack (2026-11-17)

Owner, 2026-11-17: "operator converters and training operators is very important." Goal: learned Neuralese
compaction operators trained and in use. Intent: AGENTS.md "Neuralese, meta-learning and adaptation";
`LEARNED_COMPACTION_OPERATORS.md` (2026-11-03 design, v2 seeded spans) and `COMPACTION_OPERATOR_CONVERTERS.md`
(case formats). The learner side is the in-graph pass-through (`NEURALESE_PASSTHROUGH_20261117.md`).

## 1. Inventory: every component of the operator mechanism

| # | Component | Where | State today | Current / stale |
|---|---|---|---|---|
| R1 | Runtime compaction sites | `ts-host/src/native/agent.ts` `compactRef` (562-584): compact-value on cut tool outputs (2202-2207), compact-params on cut argument listings (`views`, 591-608), compact-history on a `compact_history` note (2211-2216) | Pluggable `compaction` (`crisp` default, `nl`; agent option, `neuralese.compaction`, env `NATLANG_NEURALESE_COMPACTION`). `crisp` everywhere: no run, held eval or collector sets it | current code, never exercised |
| R2 | Frames shown to a consumer | `ts-host/src/native/operator-frames.ts` (pinned to the Python registry by `test_learned_operators.py`) | `Compacted value: <state>` etc., added *beside* the text cut (the full value stays reachable) | current |
| R3 | Operator client | `ts-host/src/model/native-operator.ts` `serverCompactor`, wired by `native-state-client.ts:46-47` when a server advertises `operator/1` | Records a `trace` `operator` event only: **no `block_write` graph node with `source_capture`, no RL capture of the operator's tokens** | stale against the pass-through (its states cannot be trained) |
| R4 | The overlapping `view` path | builtin `ts-host/src/builtin/view.nl`, `neuralese/view.ts` `serverViewer`, `training/neuralese/natlang_neuralese/view.py`, `POST /v1/neuralese/view` (`serve/http.py` 371-) | Owner decision 2026-10-09 "one summarizer family": `view(value, instructions?)` *is* the promptable summarizer. agent.ts uses the compact-params operator when a compactor is wired, else `view`. The view write is a template write (reply forced to `return_result`, value written as a native produce span, temperature 1, RL-captured `source_turns`) | current, a second implementation of the same concept |
| E1 | Python native engine | `serve/engine.py` `native_operator` / `_native_operator_impl` (916-1000), `serve/http.py` 307 | Greedy generation after `site + produce_open`, role tokens banned except `produce_close`, span up to `seed_cap`. Returns the state; **no `source_capture`, no RL capture, greedy (not the run's sampling law)**. The close marker is effectively unlearned (FALLOW C6: log-prob about -5 after 23-34 tokens), so untrained spans run to the cap | stale (see §2) |
| E2 | SGLang actor | `serve/sglang_actor.py`: `SglangEngine(Engine)` inherits `native_operator`, so it runs on the resident Python native model, not SGLang; `operator/1` is advertised by the shared `NATIVE_CAPABILITIES` (`http.py` 43-47). `_sglang_capture(ids, start, end)` (1194-1210) already captures any span *with a prefix* | parity by inheritance, slow, same defects as E1 | stale |
| E3 | Managed llama-server (fork) | `../llama.cpp-neuralese` `server-context.cpp` 7724: capabilities `chat, guidance, guidance.check, score, store, store.native-state` (+ `rl.capture` when it produces) | No `operator/1`, no `view`, **no `native.text-write`**: states only from sampled produce spans; a grammar-forced span works (wllama smoke) | gap, not in `docs/ENGINE_PARITY.md` |
| E4 | In-page engine (wllama) | `ts-host/vendor/wllama`, same release as E3 | as E3 (grammar-forced span verified by `scripts/wllama_smoke.mjs`) | gap, not in `docs/ENGINE_PARITY.md` |
| P1 | Registry and text-init rows | `training/neuralese/natlang_neuralese/model/learned_operators.py` (v2: compact-value 512, compact-params 384, compact-history 1024, extract 256 span caps; guidance) with `OperatorRows` (embedding rows per prompt token, initialized from the pinned embedding) | Tests pass (13, `test_learned_operators.py`, `test_operator_graph.py`, `test_operator_seeds.py`, ledger unit opr-tests-1, CPU). Rows are trained by nothing; a checkpoint without rows serves the text-initialized form (= the text prompt) | current; rows are an optional parameterization (§3.5) |
| P2 | Renderer | `data/operator_graph.py` (`OperatorGraphRenderer.site/seed/_producer`, text-cut siblings) | Renders the v2 graph format of the retired native-graph trainer (`natlang.neuralese-native-graph/1` + `operators`), whose consumer was `train/native_state_sft.py` (deleted). The `site()` method is still used by E1 | site rendering current; graph output stale (no current learner reads it) |
| P3 | Seeds | `data/operator_seeds.py` `seed_for` (gold / checked / `mechanical:*`) | current, tested | current |
| D1 | Converters | `scripts/neuralese_data/operator_cases_*.py`, `operator_cases_common.py`, `data/operator_cases.py`, `swezero_eval.py`, `operator_stage_sources.py`, `build_operator_graphs.py` | deleted in 7a1cbe75f (restorable from `7a1cbe75f^`) | see §2.2 |
| D2 | Cases | `/mnt/external/natlang-compaction-operators-20261103/{cases,broad-cases,graphs,ablation,swezero-eval}` | see §2.2 | see §2.2 |
| L1 | SFT learner | `scripts/neuralese_data/exp0_sft.py` (per-call chains; pass-through `imports` + `sources` in progress by the sibling, uncommitted) | can train an operator call as an ordinary chain (site -> seed reply) today; the consumer's gradient into the operator needs the pass-through | pending sibling |
| L2 | GRPO learner | `scripts/neuralese_data/grpo.py` + `lower_rollout_turns.py` (`write_sources` reads `block_write.source_capture`: `{ids[:stop], start, stop}`, any token prefix) | Trains every RL-captured call of an episode. An operator call is not RL-captured today (R3/E1), so it would not be trained | needs R3/E1 change |
| M1 | Use ablation | `eval/neuralese_ablation.py` (actual / shuffled / zeroed / text-cut consumer NLL) | deleted in 7a1cbe75f | to rebuild on the fast learner forward (§4) |
| M2 | Held task success | `rl/held_eval.py measure` | runs the collector runtime with `compaction` unset (`crisp`) | needs a `--compaction nl` arm (§4) |
| X1 | Held exposure | `held_eval.exclusion_ids()`, `ops/held-v3/exposure/collect_exposure.py` | see §2.3 | see §2.3 |
| K | Call-kind registry | `training/model-call-kinds.json` 47-49 (`operator-write`) | still describes the retired v1 ("K learned slot positions") | stale text; rewritten with the v3 write |
| F | Fallow rows | `FALLOW_REGISTER.md` C7 (runtime hooks), C8 (operators), C6 (native writer: close marker), C2 (soft states) | | updated with this work |

How often the runtime reaches a site today (offline, 60 random grpo-v3/run1 rollout files, 480 episodes, 7,420
calls): **100 episodes (21%) contain a cut-off tool output** (740 cut tool messages in their calls), i.e. a
compact-value site; **7 episodes (1.5%) call `compact_history`**; **0 cut argument listings** (compact-params sites do
not occur in the RL pool's calls). 117 episodes carry 397 Neuralese `block_write`s (programs' own text writes).
compact-value is the operator real rollouts can train; compact-params and extract need case data. A cut is recorded
whenever a tool result's display differs from its full rendering (`runtime.ts` `show`/`showValue`, 1401-1421), so
many sites are structural cuts of modest values: in 40 sampled files the cut notes of final calls read e.g. "280 of
315 characters not shown" or "9 of 10 items not shown". The full values are not archived in the rollouts, so their
size distribution is unmeasured; the site window bounds the learner's cost either way.

## 2. Current vs stale

### 2.1 Runtime surface, tokenizer and template

- **Tokenizer, template and markers: current.** Every graph receipt records tokenizer fingerprint `133504d4…`
  (which covers the chat template, sha256 `f45005c5…`) and role ids 125033-125036. Today's `provision_role_tokenizer`
  on the pinned snapshot gives the same values, and so does the LFM3 entry in `common/line_backbones.py`. grpo-v2
  rollouts from 5ff07638 record the same fingerprint. (The 5ff07638 checkpoint's own metadata was not opened.)
- **Rendering: current.** Today's `OperatorGraphRenderer`, with today's tokenizer on CPU, re-rendered 3 cases from each
  of the 23 sources (69 in all) byte-identical to the stored graphs: token ids, weights and operator declarations. The
  registry hash differs only for extract (`82f14bd41` added fields whose defaults leave it unchanged). The renderer
  has no driver on main (`build_operator_graphs.py` was deleted).
- **Stale: the consumers of that rendering and the runtime write.** The v2 graph format's only trainer is deleted.
  The serving write (E1) has neither RL capture nor `source_capture` (§1, §3.1).

### 2.2 Converters and cases

The cases are text `natlang.port-record/1` records (sources, writer instructions, consumer context, target, split,
groups, seed in `meta`). They do not depend on the tokenizer or template, so they are **current as data**; only
their rendering is stale. Train / test counts (from the manifests), with seed provenance:

| Operator | Source: train / test | Seeds |
|---|---|---|
| compact-value (≈37.7k train) | tool-digest 9,249 / 1,911; web-search-r1 15,000 / 509; agent-tool-slots-repo-qa 4,518 / 282; web-miroverse 3,997 / 155; web-openseeker 3,032 / 27; web-openresearcher 1,198 / 9; web-sds 428 / 6; web-webshaper 249 / 1 | checked, except web-search-r1 (mechanical answer lines) |
| compact-params (≈20.6k) | s1-repo-qa 6,201 / 297; s1-qa-long-context 4,000 / 300; s1-qa-narrative 3,805 / 300; s1-qa-multihop 3,000 / 300; s1-qa-trivia 3,000 / 300; s1-qa-mcq 565 / 246; s1-qa-narrative-long 0 (all rejected by the 8-source / 110k limits) | mechanical (answer lines, overlap sentences) |
| compact-history (≈29.9k) | s1-swe 29,698 / 1,000; bgkit-compaction-v2 173 / 0 | mechanical history outline |
| extract (≈30k) | broad-hotpotqa 5,889 / 222; broad-multi_news 6,049 / 300; broad-squad 6,049 / 293; broad-musique 5,361 / 210; broad-fetaqa 4,443 / 190; broad-2wiki 1,561 / 77; broad-qmsum 648 / 300 | gold (dataset spans), checked (multi_news ROUGE oracle) |

- `graphs/` (682 MB) holds the v2 native-graph renderings. They match today's tokenizer byte for byte (§2.1), but their
  format has no trainer and puts the seed as a span after `produce_open`, not as a reply. Rows are re-lowered from
  the cases (§5 step 2); the renderer's `site`/`seed` are reused.
- The converters need restoring only to add or refresh cases. Their helpers are all still on main (`records.py`:
  `exact_refs_from`, `group_key`, `seal`, `license_`, `source`, `strip_*`; `common.py`: `Reject`, `parquet_rows`;
  `bgkit.py`; `common/paths.py`). Restore them from `7a1cbe75f^` when a case set is regenerated.
- compact-history needs a decision before training. Its cases supervise a separate operator call that writes a
  history outline. v3 instead makes the history state the model's own `compact_history` note captured in context,
  so these cases fit only as "operator call writes the note" rows. They train that if the note is kept as the
  operator's reply; the RL pool has `compact_history` in only 1.5% of episodes.

### 2.3 Held exposure

- **Not yet safe to train.** The cases were filtered at build time against the older protection manifest
  (`acquired-connected-source-protection-20261017…`, which dropped 6,245 train rows), never against held-v3/v2.
- `held_eval.exclusion_ids()` filters RL pool rows by task id and `curriculum.split_group` only, so it cannot catch
  these cases.
- The case `split_groups` use the same namespaces as held tasks' `source_groups` (`2wiki-q:`, `triviaqa-q:`,
  `hotpotqa-q:`, …). held-v3 contains 2wiki, musique, triviaqa, narrativeqa and searchqa tasks, and so do
  broad-2wiki, broad-musique, s1-qa-trivia, s1-qa-multihop and s1-qa-narrative.
- **The row lowering (§5 step 2) therefore drops every case** whose `split_groups`, upstream id or input text chunks
  hit held-v3/v2 (the same group and text checks `held_eval verify` uses). The actual overlap count is pending: scan
  unit `opinv-scan3` (`/tmp/claude-1000/opinv/scan.out`) is still running on the saturated `/mnt/external` disk, and
  s1-swe is not yet scanned.
- The cases are not in `training/neuralese_corpora.json`.

## 3. Training design on today's stack

### 3.1 The operator write: an ordinary call whose reply is captured in its own context

Today's write paths each break one requirement:

- E1's greedy span after `produce_open`: not sampled under the run's sampling law, not RL-captured (GRPO cannot
  train what it writes), and the length is the cap because the close marker is unlearned (C6).
- A standalone text write (`native_write_text`, what every program producer uses today): RL-captured and trainable,
  but the state is computed from the text alone at positions 0..n; its rows never attend to the value, so the state
  carries exactly what the text says and nothing more. That is not compaction of the value.

**Decision (v3).** An operator write is two steps on every engine:

1. **The operator call:** one ordinary chat call at the operator's site (`operator_site_messages`: the prompt as
   system text, the instructions and the value as user text; history: the conversation plus the prompt as the last
   user message). The reply is plain text, sampled under the run's sampling law (temperature 1, top-p 0.95), with
   `max_tokens` = the operator's span cap. It ends with the ordinary end of turn, which the model already emits, so the
   unlearned close marker plays no part. The call is RL-captured like every other call.
2. **The in-context capture:** the state of the reply's tokens is captured *in its own context*:
   `ids = site + [produce_open] + reply`, rows `[len(site) + 1, len(ids))`. The rows attend to the whole value, so
   the state can carry more than its text. The write is recorded as a `block_write` graph node with a
   `source_capture` whose segments are `site (not producing), produce_open, reply (producing)`, with `payload_start`
   and `payload_stop`. That is exactly the form `lower_rollout_turns.write_sources` already reads (any token prefix):
   the pass-through computes the state in-graph from the site and reply and sends the consumer's gradient into the
   weights that computed it, through the whole site forward.

Text initialization is literal: the operator prompt is text in the real rendered call. The learned prompt rows of
`learned_operators.OperatorRows` are an optional later parameterization: the backbone trains; if a soft prompt is
wanted it is the existing system-prompt bank (`softenText`, as `view.ts` already uses), not a second mechanism.

**One summarizer family (owner decision 2026-10-09).** compact-value and compact-params are `view(value,
instructions)` at the runtime's sites (the instructions are the receiving call's purpose), so they use view's body
(`builtin/view.nl`) as their site prompt and one write path; the separate VALUE/PARAMS prompts and the
`/v1/neuralese/operator` endpoint retire when the v3 path lands (version bump, MODEL_FACING_CHANGES entry).
compact-history needs no extra call: the model's own `compact_history` note *is* the history compaction; its
state is the note's in-context capture at the compaction call (the note's rows attend to the whole conversation).
extract is a program pattern (a natlang function returning `Neuralese<string>`); it uses the same in-context
capture when the general producer path adopts it (§3.6).

**Engines (one serving feature).** The capture step is one protocol request. A contextual capture already exists:
`native.init-body` (`engine._native_init_body_impl`, engine.py 1216-, `nativeTextOperation('init-body')` in
`native-state-client.ts`) captures supplied text at its actual rendered position inside a call, with produce markers
around it and a full `source_capture`. The operator capture extends that one mechanism (the reply as the
assistant turn of the site messages) rather than adding a third write path. One change to it: init-body re-tokenizes
the supplied text (engine.py 1258-1260); the operator capture passes the reply's *sampled token ids*, so the captured
span is exactly what was sampled (a re-tokenized reply can differ, the 60d2c1209 finding):

| Engine | Implementation |
|---|---|
| Python native | the init-body capture with the reply as the site's assistant turn |
| SGLang actor | today `native_init_body` falls back to the resident Python model (`sglang_actor.py` 328-330); the capture becomes `_sglang_capture(ids, start, end)` (1194-1210), the existing capture request, which already takes any prefix (no fork change) |
| Managed llama-server, page | a guided chat completion at the site whose grammar forces the exact token ids `<[produce_open]> <[r1]> ... <[rn]> <[produce_close]>` (grammar token syntax as in `scripts/wllama_smoke.mjs:87`), i.e. the produce span the fork already writes. This also closes the existing unrecorded gap that the fork serves neither `native.text-write` nor `native.init-body` (`server-context.cpp` 7724) |

### 3.2 Data

- **Cases** (§2.2): each becomes two per-call training rows in exp0_sft's chain format.
  - *The operator call:* site -> seed text as the reply target (gold / checked / mechanical provenance kept). Seed
    CE weight anneals 1.0 -> 0.1 (`operator_seed_schedule`, unchanged from v2).
  - *The consumer call:* the consumer context with the state imported, target = the case's answer.
    `imports = [(start, key)]` and `sources[key] = {ids: site + [produce_open] + seed, start, stop}`, i.e. the
    pass-through's format.
  - The text-cut sibling (consumer reading the runtime's text cut) stays as the ablation baseline.
- **Real rollouts.** Collection with `compaction: nl` gives compact-value sites in about 21% of episodes, plus
  compact-history notes. Operator calls are RL-captured calls of the episode and its consumers import their states
  with recorded `source_capture`: GRPO trains both with the episode's advantage, with no special case.

### 3.3 Loss

- SFT (cases): `w_seed(t) * CE(seed | site) + CE(answer | consumer, state(site, seed))`, where `state` is the in-graph
  pass-through. The consumer's loss reaches the operator's site forward (every layer below each captured row, the
  embeddings of the site and seed tokens). The seed text itself is discrete: what to write is trained by the seed
  CE (imitation) and, in RL, by the policy term.
- RL: unchanged GRPO over all calls (`-min(rho,2) A logp + k2`), with pass-through imports. The operator's sampled
  reply is trained by the episode advantage, i.e. by task success with the compacted state. The state's encoding is
  trained by the consumers' terms through the pass-through.

### 3.4 When the runtime uses an operator

`compaction` stays `crisp` by default. The arm is a model-facing variant, `learned-compaction`
(`ts-host/src/runtime/model-facing.ts`), which makes `nl` the compaction default: `NATLANG_MODEL_FACING` already enters
the held eval's series key (`held_eval.series`, `model_facing`), so `compare` cannot pair an `nl` result with a
`crisp` one, and grpo collection selects it the same way. (`NATLANG_NEURALESE_COMPACTION` stays the explicit
override; it is not in the series key, so held evals use the variant.) Default flips to `nl` when held task success with `nl` is not below `crisp`
on the same checkpoint (paired, p >= 0.05 for a drop and point estimate >= 0) and the ablation shows use (§4).

### 3.5 Measurement

1. **Use ablation (fast, forward-only, ≤15 GB):** per case, consumer-target NLL with the actual state, a state
   shuffled from another case of the same operator (derangement), the same reply text written standalone (from
   position 0, today's text write), and the text cut instead of the state. Read
   per-row paired differences with a sign-flip p and a bootstrap CI, by seed provenance. Rebuilt on the fast learner
   forward (the same `passthrough_plan` the learners use) instead of the deleted native-graph tool.
2. **Held task success:** `held_eval measure` with the collector in `compaction: nl` vs `crisp`, same checkpoint,
   2-4 samples per task; the primary statistic as for every held comparison.

### 3.6 Generalization (fix the mechanism everywhere)

The standalone text write is the shared mechanism behind every Neuralese state today. In-context capture is the
better write for any producer whose result summarizes its own inputs. The operator path lands it as an option of
`native.text-write` on every engine. Moving program producers (`return_result` of a `Neuralese<T>` function) to it is a
separate decision for the pass-through owner: their producer documents become whole producer calls (5-30k tokens)
instead of 8-91 tokens, which the learner can share with the producer call's own forward only when both are in one
packed row.

## 4. The v1 slot smoke (why it showed no row-specific information)

Two independently briefed investigators (static analysis plus one CPU probe, ledger unit `opdiag2-probe1`) found **no
implementation bug** in v1's capture, import or ablation, within this scope: capture rows and positions, the import
splice, the shuffled derangement, the zeroed construction, RoPE, dtype, caches, text-init exactness and consumer
weighting. They agree on the cause, a design property that predicts all three directions:

- **Mechanism.** v1's imported span was K slot positions whose tokens were one constant phrase per operator, repeated
  to K (`ff3ea04e4:model/learned_operators.py` 116/167, `seed[i % len(seed)]`; `data/operator_graph.py` 110-111). Token
  ids and input embeddings were identical for every row. Row-specific content could reach the slots only through
  attention, and 6 of LFM3's 12 attention layers have a 256-token window that cannot see a 2-8k-token value.
- **Measured.** On base LFM3 (bf16, CPU, 8 rebuilt compact-params sites), the share of slot hidden-state energy that
  varies across rows is 0.000 at the embedding and layers 1-2, 0.014-0.056 in the middle layers and 0.11 at the last
  layer. About 95% of the imported state is the same in every row.
- **Predicted directions.** Shuffling swaps only the ~5% that differs, so shuffled ≈ actual. Zeroing removes the
  shared 95% as well, which is out of distribution, so it hurts whether or not content is used. The text cut carries
  real content, so it scores best. Spans whose tokens differ per row separate cleanly (written text +0.91, v2 seeded
  extract +1.51 nats).
- **The rows could detect use.** compact-history mostly could not (the consumer keeps the task and recent steps; all
  four conditions fall within 0.556-0.568). compact-params and extract could: text-cut beats actual by 0.24 and 1.56.

**Not checked:**
- the deleted bf95 checkpoint and its producer adapter (from the code, it acts identically on identical slot
  embeddings);
- row variance in K/V space rather than in the residual stream;
- the original smoke graphs (deleted; sites rebuilt from the v2 graphs).

**Today's code does not have the defect:** v2 generates an input-dependent span, and v3 samples a reply.

**Lesson for v3, from the same measurement.** A state's row-specific content comes mostly from its own tokens. In-context
rows add what the full-attention and conv layers carry from the value (the window-256 layers see only the reply's last
256 tokens of context). So the ablation must also compare the in-context capture with a standalone write of the same
reply text: if they do not differ, the in-context capture buys nothing and costs a site-length source document.

## 5. Plan

1. This note (inventory, staleness, design, diagnosis).
2. **Cases -> learner rows.** A lowering `scripts/neuralese_data/operator_rows.py` from the port-record cases to
   exp0_sft's per-call rows (operator call row; consumer row with `imports` + `sources`; text-cut consumer row for the
   ablation), tokenized with today's template and tokenizer, with held exclusion applied. It replaces the retired
   graph renderer output (`OperatorGraphRenderer` keeps `site` and `seed`). Restore only the converters whose cases
   are missing or stale (§2.2); regenerate under the ledger (CPU, low priority); publish rows as corpora with
   SHA-256 manifests in `training/neuralese_corpora.json`.
3. **Runtime and engines (v3 write).** ts-host compactor: the operator call through the ordinary chat path
   (RL-captured), then the contextual capture, recorded as a `block_write` with `source_capture`;
   compact-value/compact-params at view's body (one summarizer family); compact-history from the model's own note;
   a site floor of 800 characters (the case floor) so structural cuts of small values do not spend a call. Python
   and SGLang: the contextual capture of a reply (SGLang through `_sglang_capture`, no resident native model). Fork
   and page: the grammar-forced capture in the ts-host client. Targeted tests per engine; `docs/ENGINE_PARITY.md`
   updated; `MODEL_FACING_CHANGES.md` entry; skills reference if the frames change. The `/v1/neuralese/operator`
   endpoint and E1's greedy span retire when the new path lands.
4. **Eval.** The use ablation on the fast learner forward (actual / shuffled / text-cut, by provenance) once the
   pass-through's `passthrough_plan` is on main; the held `--compaction nl` arm in `held_eval` (recorded in the
   series identity).
5. **Training** once the pass-through lands: SFT on case rows (operator + consumer through the state), then GRPO with
   collection in `compaction: nl`. Start measurement: the ablation at step 0, then after the SFT stage; held success
   `nl` vs `crisp` on the trained checkpoint.

Interface needs from the pass-through (via the supervisor): none beyond what `lower_rollout_turns.write_sources` and
`exp0_sft` `imports`/`sources` already accept (a source document with any token prefix, `{ids[:stop], start, stop}`).
The cost to watch: operator source documents are the site (value + prompt, up to the site window) rather than 8-91
tokens.

## 6. Progress (2026-11-17 evening)

Coordinator approval: the v3 design as above. The held filter is mandatory and matches split groups by name. Fork/page
parity is closed in this stream. Operators join the running GRPO at a resume once the ablation and the held on/off
check pass; the variant means a new Pop base.

- **Registry v3** (`learned_operators.py`): compact-value and compact-params are `view` at the runtime's sites (view's
  body, `view.view_site`, placement `view`). Tests: `test_learned_operators.py`.
- **Rows** (`scripts/neuralese_data/operator_rows.py`):
  - Each case gives an operator chain (site -> seed reply), a consumer chain with `imports`/`sources` (the in-context
    document: site + reply up to its content + `produce_open` + content), and, for test cases, a text-cut chain.
  - The held closure covers held-v3 and held-v2: exact group names, task, source and upstream ids, and 10-word
    shingles of held inputs. A shingle shared by two held strings is boilerplate and does not count; a text hit needs
    25 shared shingles or half of a short held string. The smoke dropped 3 of 4 s1-qa-mcq cases on the Gutenberg
    notice; the shingle rule now excludes that boilerplate.
  - Tests: `test_operator_rows.py`, including `passthrough_plan` on a real row.
  - The full build (compact-value, compact-params, extract) is queued as ledger unit `oprows-build-2` (low priority,
    output `/home/werg/data/natlang-operator-rows-20261117-v1/`). Its `manifest.json` records the held-drop counts by
    kind and split.
- **compact-history cases are not lowered.** They train an operator call with HISTORY_PROMPT, but v3's history state
  is the model's own `compact_history` note captured in its context. Their 30k cases × ~30k tokens would train a site
  the runtime never runs. They stay on disk for a note-capture rendering later.
- **Engines** (commit d306de499):
  - `native.capture` (`POST /v1/neuralese/native-states/capture`) on the Python engine (`native_capture`) and the
    SGLang actor (`_sglang_native_capture`, the fork's capture request).
  - The tiny-model test showed that a 2-layer conv+attention state sees its context only through the conv window: the
    §4 mechanism again. The test varies a token inside the window.
- **ts-host** (commit d306de499):
  - `compactRef` runs compact-value/compact-params as an operator call (`<call>#operator-<n>`, view's site,
    `max_tokens` 2x the cap), then `port.capture` on its sampled ids. A `block_write` node carries the
    `source_capture`.
  - 800-character floor; model-facing variant `learned-compaction`.
  - Tests: `native-operator-compaction.test.mjs` (3 pass); `model-call-kinds.json` gains `operator-call`.
- **Ablation** (`scripts/neuralese_data/operator_ablation.py`): arms actual / standalone / shuffled / text-cut on the
  fast forward. It runs when the rows exist and a ≤15 GB GPU slot opens.
- **Still open:**
  - The fork/page capture: a client-side grammar-forced span after the same site messages. Its shape is decided; it
    needs the content split (`</think>` and end ids from the server's tokenizer) and a live check against the
    natlang-b12081 runtime with the natlang-lfm3-5ff07638 GGUF. Recorded as the open row in `docs/ENGINE_PARITY.md`.
  - compact-history as note capture.
  - SFT, then GRPO with the variant.
  - Held on/off.

## 7. Step-0 ablation, fork parity, held filter audit and the stage command (2026-11-17 night)

- **Step-0 ablation** (5ff07638, seeded states, 64 test consumers per source;
  `/home/werg/data/natlang-operator-rows-20261117-test-v1/ablation-step0-5ff07638.json`).
  - The in-context state beats a shuffled donor by 0.7-2.7 nats/token on every source with power.
  - It beats the same span written standalone by 0.5-2.6 nats/token on 17 of 21 sources (p<0.05), so the in-context capture
    carries more than its tokens.
  - It beats the text cut on compact-params and compact-value. It is about even on extract, whose documents are
    short, so their "cut" is the whole text.
  - Caveat: these are seeds, not the model's own operator replies, and mechanical seeds bear the answer.
- **Fork/page capture parity** (natlang-b12081 Q8_0 vs Python BF16, same ids): same 38 tensors, relative L2 median
  0.025 and largest 0.048, the Q8 floor. The `docs/ENGINE_PARITY.md` row is removed. The live run found two defects,
  both fixed:
  - a grammar together with an RL capture is refused;
  - replies cut off inside their reasoning were captured. They are now refused, and operator calls get 2048 tokens of
    reasoning room.
- **Held filter audit.** s1-qa-long-context dropped 609 of 4,000 train cases (15.2%; the source's whole reject rate is
  611 of 4,300 = 14.2%, plus 2 over the token cap).
  - 447 of the drops share passages of one held triviaqa task's input: the Wikipedia "Antarctica" page, whose 8,257
    shingles are shared 59-101 at a time by chatqa2_long_sft documents quoting the same article. 129 share whole
    paragraphs of two held musique tasks (79 of 79 shingles, i.e. the full held paragraph). The rest share passages
    of squad-based held tasks.
  - The shared-shingle quantiles are 21 (half of a short held string), 71-185, and 4,829 at most. None of the
    inspected matches was boilerplate. The boilerplate exemption only removes shingles that two or more held strings
    share; here the shared text is genuine held input.
  - Decision: keep dropping. A held passage in training is the worse error.
- **Stage command:** `/home/werg/data/lfm3-rl-dgx-20261029/ops/operators-v1/operator-stage.sh`.
  - `merge`: rows -> one `--chains` pickle and a `--val-chains` pickle of 32 test consumers per source
    (`operator_rows.py merge`).
  - `sft PIN CKPT`: operator-rows SFT continuation through `grpo-v4/launch-sft.sh` (LR 3e-6, one epoch).
  - `held-base PIN`: Pop 4-sample base of 5ff07638 under `NATLANG_MODEL_FACING=effect-receipt,learned-compaction`.
  - `grpo PIN`: GRPO from the SFT checkpoint with the variant, new WORK `ops/grpo-v4/run2-operators`, held base above.
  - The grpo-v4 launchers are used unchanged. The variant enters their ledger env through a generated copy that adds
    one variable to the env line.
  - The pin must contain main at or after this commit.
