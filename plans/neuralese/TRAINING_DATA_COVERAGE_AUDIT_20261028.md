# Training-data coverage audit: model-facing call kinds (2026-10-28)

Goal: every request the runtime sends to the executing model at inference has matching training data: a supervised
target for what the model produces (SFT, `own-outputs/1`), an RL action with gradients through its real graph
dependencies, and a training rendering byte-identical to what the runtime sends. Trigger: the `decide` readout
(owner frustration that such gaps keep appearing). `decide` itself is being fixed by its own agent; it is listed
here only as a cross-check.

Standing guard: `training/model-call-kinds.json` declares every call kind with its SFT, RL and parity coverage, and
`tests/neuralese/test_model_call_kind_coverage.py` fails when a model call site in `ts-host/src` (an agent turn, a
scoring request, a request to a model endpoint, a runtime-authored natlang call) or a route in `serve/http.py` is not
declared there, when a declared kind is no longer issued, or when a column has no evidence, owner or reason. A gap is
allowed only as an explicit entry pointing at its section below.

## 1. Inventory (derived from code)

Call sites found by the guard's scan of `ts-host/src` (line numbers on branch `worktree-agent-ac2db8bd04f632f22`):

| Kind | Site | Endpoint | What the model produces |
|---|---|---|---|
| tool-loop-turn | `native/agent.ts:1462` (`this.driver(turnRequest)`) | `/v1/chat/completions` | reasoning, tool calls, typed answer; includes opening, template readouts, repair turns after rejected actions, protocol-failure retries, compaction requests, history with preserved thinking (LFM3 `keep`) |
| runtime-authored-call | `runtime/runtime.ts:394` (`refinement:holds`, no scoring driver), `runtime/iterate.ts:156` (builtin `progressJudge`), `native/runtime.ts:1432` (bash `ask`), `:1481` (`delegate` tool reducer), `:1514` (`nl@python`), `cli/main.ts:772` (`natlang ask`), `calls/judge.ts:48` (specializer `compareBehaviors`) | `/v1/chat/completions` | same as tool-loop turns, with runtime-written instructions |
| decision-readout | `native/agent.ts:1159` (`decide({messages})`) | `/v1/neuralese/decide`, `/v1/neuralese/decide_many`, `/v1/natlang/score` | nothing generated: every allowed JSON value is scored as the reply |
| refinement-judge-scored | `native/refinement.ts:92` (`scorer({messages})`, `JUDGE_SYSTEM_PROMPT`) | same scoring endpoints | nothing generated: `true`/`false` scored |
| careful-review | `native/agent.ts:1598` (`review_write` fork) | `/v1/chat/completions` | reason + decision via `review_write` (opt-in `model.review`) |
| view-write | `neuralese/view.ts:57` | `/v1/neuralese/view` | template writes of a value's view |
| native-forced-write | `model/native-state-client.ts:113` (`write-text`, `init-body`) | `/v1/neuralese/native-states/*` | forced text encoded to native state (no sampled output) |

Not model calls: LoRA mixture/coefficient selection is a declared deterministic binding
(`model/native-coefficient-policy.ts`, adapters ride on recorded turns); `/v1/neuralese/guidance/check` and
`/v1/neuralese/render` are crisp diagnostics with no caller; grad/optim/adapters/collect/info are learning and store
operations. Guided generation masks or forces tokens inside tool-loop turns; RL logs the exact masked law
(`x_natlang_rl_top_k`), SFT targets the produced text. Teacher-side drivers (`teacher/model-judge.ts`,
`teacher/refine-repair.ts`, `tune/repairs.ts`, `optimization/` reflection) do not run the student at inference.

## 2. Coverage matrix

Columns: (a) captured in recorded trajectories; (b) survives conversion into LFM3 joint-SFT rows; (c) own-outputs
objective weights the model-produced part; (d) whole-graph RL action with gradients through producer/consumer
dependencies; (e) training rendering equals inference rendering; (f) examples in admitted sources (line matches over
the LFM3 mixture's 113 scored teacher sources, about 40 GB); (g) exercised by the development packet
`natlang-dev-task-packet-v1-20261028`; (h) JOLT+ privileged teacher / score centering can score it.

| Kind | (a) | (b) | (c) | (d) | (e) | (f) | (g) | (h) |
|---|---|---|---|---|---|---|---|---|
| tool-loop-turn | yes (linked, `agent.ts:1489-1511`) | yes (renders, `student_rerender`, graph views) | yes (`own_output_supervision.py:277-311`; forced generation prompt 0 since `dd856b7a5`) | yes (sampled positions, `rl/native_replay.py:298`) | **no for program-header rows recorded after 2026-10-25**: stale piece table (gap G1, fixed) | all rows | 128/128 cases | yes |
| runtime-authored-call | yes, as ordinary calls (`internal: true` only disables specialization, `kernel.ts:383,598,755`) | yes, if present | yes | yes | as tool-loop | progress-judge verdict lines: FILL; `refinement:holds`: 0 | progress judge in iterateOn cases without a bound (40 cases use iterateOn) | yes |
| decision-readout | recorded and linked only under a native recorder (`agent.ts:1168-1171`) | **no** (no generated target) | **no** | **no** (`native_replay.py:184-186` refuses; `tune/rollouts.ts:160` skips the rollout) | untested (no fixture holds `DECISION_SYSTEM_PROMPT`/`decisionPrompt`) | teacher collection never scores: decisions appear only as generated turns (`teacher/collector.ts:1315` driver has no `decide`) | 0 (no packet function declares `readout: decision`; no profile sets `finite-returns`) | no (no sampled positions) |
| refinement-judge-scored | recorded as a decision turn under a native recorder (`neuralese-server.ts:672-677`) but **never linked** to a graph node | exact-oracle rows exist (`refine-judge-exact-20261009-v1`, 1,020 train rows; LFM3 render done, not registered) but the LFM3 mixture's answer-only drop rule removes them (gap G3) | yes when present (target `</think>false<|im_end|>` after the forced `<think>`; matches the scorer's empty-thought continuation) | **no**: silent drop in collector episodes, refusal in recorded objectives (`native_recorded_graph.py:36-37`), rollout skip only when the judge uses the wrapped driver (`rollouts.ts:144-145`) | prompt parity tested at message level (`tests/test_refine_data_judge.py:116`) | `Is<` 0, `refine(` 0, `refinement-unsatisfied` 0 lines in teacher sources | 0 | no |
| careful-review | **no graph node, no invocation id** (`agent.ts:1598`) | none | none | none: collector episodes with a review are rejected whole, rollouts skipped | none (no Python counterpart; program-header sends `tools: []` while requiring `review_write`) | 0 `review_write` lines | 0 (opt-in, no profile enables it) | no |
| view-write | template turns captured server-side (`source_turns`), only `block_write` graph nodes | none | none | none | server-side | 0 | 0 | no |
| native-forced-write | block metadata (`native_source_capture`) only, no recorded turn | n/a (forced) | n/a | **external root** (`rl/memetic_program.py:543-560`); only memetic SFT links a producer | XML write sites tested (`test_lfm3_format.py`) | n/a | 0 | imports only |

Applications that the runtime serves use these kinds: 15 application functions declare `readout: decision`
(`applications/{scheduling,workflow,wiki,nldb,logs,fusion-planner,...}`) and 17 application sources use refinement
types `Is<T, P>`, so decision readouts and scored refinement judges run in production paths whose training coverage is
zero.

## 3. Gaps with evidence

<a id="g1-stale-piece-table"></a>
### G1. Program-header renders keep runtime instructions the runtime no longer sends (fixed)

Program-header rendering (`data/prompt_rendering.py`, used by every training render, graph view and by serving when a
client sends a full prompt) removes generic runtime instructions by matching them against
`training/prompt-pieces/pieces.json`, exported from ts-host `promptPieces()` at `18dda11` (2026-10-11). ts-host
changed the interpreter prompt since (`7b027d2fb` 2026-10-23, `38f10624b` 2026-10-25 added
`SEMANTIC_CALL_COMPLETION_PROMPT`, `614ad5e48` 2026-10-26), and the table was never re-exported. Measured with the
live ts-host `renderSystemPrompt` over the table versus over the current pieces (`ts-host/dist` at 06219535c): the
current `interpreter` and `interpreter/depth-limit` pieces are missing from the table, and the program-header
rendering of the current `TOOLS_PROMPT` keeps **15,666 characters** of generic instructions through the table versus
**659** at inference. Every program-header or whole-trajectory row rendered from a trajectory recorded with the
runtime of 2026-10-23 or later therefore trains on a system turn the served model never sees.

Exposure today: the 34 LFM3 renders on disk (7,381 program-header rows) come from sources recorded on or before
2026-10-15 and keep no generic interpreter text (`generic_interpreter_text_kept` 0, per-row scan); the teacher-twin
graphs are rendered `full` (G7). The defect would have hit every later render (Luna 2026-10-26/27, Qwen3.6
batches, tuning-lane rows) and full-prompt clients of a program-header server.

Fix (this branch): `pieces.json` regenerated with `export-prompt-pieces.mjs --history` (current pieces plus all
historical revisions); `ts-host/test/prompt-pieces-export.test.mjs` fails when a current piece is missing from the
table or when the table renders `TOOLS_PROMPT` differently from the runtime. The rendering-contract identity
(`data/rendering_contract.py`) changes with the table; it is reported in `/v1/neuralese/info` and startup logs, not
used to refuse a checkpoint, so served models are unaffected. `check_prompt_against_ts_host.py` (broken since
`7b027d2fb` deleted the module it imported) is repaired and now counts generic paragraphs left in headers.

<a id="decision-readout"></a>
### Decision readout (cross-check; owned by the `decide` agent)

Confirmed independently: no SFT target, no RL action, no rendering-parity fixture, no development-packet case, no
teacher example in the scored form. Additional sites the fix must cover, beyond `agent.ts decisionReadout`:
- `tune/rollouts.ts:145,160` skips every rollout whose root used a scorer (`decided`), and its docstring (lines 9-10)
  calls that intended; every tuned program with a decision or a scored judge yields zero rollouts.
- `neuralese/text-init.ts:34` captures the decision request to build an init-body site: whatever form decisions
  take must keep this capture equal to the served request.
- Scored refinement judges share the scorer (next section); a mechanism that makes decision readouts actions must make
  judge verdicts actions too.
- Teacher collection never scores (`teacher/collector.ts:1315`), so recorded teacher data contains decisions only as
  generated turns: the scored prompt (`DECISION_SYSTEM_PROMPT` + `decisionPrompt`) never appears in training data.

<a id="refinement-judge-scored"></a>
### G2. Scored refinement judges are invisible to training (handed to the `decide` owner; data part to the mixture owner)

At inference on a native server every refinement predicate without a crisp checker (`RefinementChecker.decide`,
`refinement.ts:228-255`, mode `crisp` falls back to the judge) is judged by `decisionJudge` (`refinement.ts:84-97`):
one scoring pass over `true`/`false` under `JUDGE_SYSTEM_PROMPT`. The verdict decides whether the call's return is
accepted, i.e. it changes the episode's outcome.
- Recording: under a native recorder the neuralese driver records it as a `decision` turn (`neuralese-server.ts:672-677`),
  but nothing calls `linkNativeTurn` for it, so `nativeTurns()` returns it without a graph binding
  (`model-recording.ts:141`).
- RL: collector episodes silently omit it (no turn, no `model_turn` node): the verdict is trained as environment.
  Recorded objectives (`train()`, tuning) refuse the whole recording (`native_recorded_graph.py:36-37`). Tune rollouts
  skip the root only when the judge runs on the wrapped driver; a separate `refinements.judge` model is not noticed
  (`rollouts.ts:144-145`).
- Teacher data: the collector cannot score, so judges ran as `refinement:holds` tool-loop calls
  (`runtime.ts:386-397`, `JUDGE_CALL_INSTRUCTIONS`), a different prompt and protocol than inference uses.
- SFT data: `refine-judge-exact-20261009-v1` (1,020 train rows, labels from crisp checkers, prompts byte-equal to
  `decisionJudge`, `tests/test_refine_data_judge.py`) is rendered for LFM3 (`runs/lfm3-renders/natlang-native-turns-refine-render-20261011-v1`,
  queue status `rendered`) with a correct target (`</think>false<|im_end|>` after the forced `<think>`, the same tokens the
  scorer's empty-thought continuation scores), but see G3. Its 93 predicates are crisp-checkable formats; the runtime
  sends the model exactly the predicates without a crisp checker (semantic ones), which no source covers.
Fix direction: link judge scorings into the graph as `readout: 'decision'` model turns (the call's trace and
`callId` are in the checker context) and give them the action treatment the `decide` fix gives decision readouts;
semantic-predicate labels come from exact oracles on training worlds (a predicate whose truth the world's oracle
fixes) and from RL outcome credit, not teacher generation.

<a id="g3-mixture-drops-runtime-kinds"></a>
### G3. The LFM3 joint-SFT mixture policy drops runtime-issued answer-only calls (handed to the mixture owner)

`training/neuralese/mixtures/lfm3-joint-sft-mixture-policy-v1.json` drop rule 1 ("no natlang program structure
(answer-only, plain QA or chat)": `evals`, `typed_natlang_calls` and `neuralese_edges` all 0) removes every judge row of
`refine-judge-exact` (a system+user prompt and a one-word reply: no eval, no child, no edge) and would remove any
future decision-readout or judge row the same way. The rule is meant for generic QA; runtime-issued call kinds need
an explicit exemption (e.g. a `runtime_call_kind` feature from the source row's kind with its own tier). The
mixture is not built yet (`runs/lfm3-mixture` holds idiom scores only), so nothing has been lost.

<a id="careful-review"></a>
### G4. Careful review turns (opt-in) have no recording identity (supervisor decision)

`agent.ts:1594-1605` sends the review fork without `invocation_id`, never calls `modelTurnNode`, and keeps the audit
only in `this.reviews` (appended, never read). Consequences: tune rollouts skip the root (`rollouts.ts:122`);
collector episodes that contain a review are rejected (`rollout_turn_names` needs an invocation id); recorded
objectives refuse an unlinked turn. In program-header mode the request sends `tools: []` while requiring a
`review_write` call. No data exists (0 `review_write` lines in teacher sources) and no profile enables review, so
this is latent. Either delete careful review (lean codebase) or record it as a `model_turn` with its own action
treatment before enabling it.

<a id="view-write"></a>
### G5. `view` server writes have no recorded turns and no runtime user (legacy sweep)

`neuralese/view.ts` (`serverViewer`) is exported but never constructed in `ts-host/src`; its server turns
(`source_turns`) reach only an optional `onSourceTurns` callback, while the graph gets `block_write` nodes. Either wire it
as recorded model turns or delete it with the legacy sweep.

<a id="native-forced-write"></a>
### G6. Forced producers are external roots in RL (RL owner)

`write-text` and `init-body` (and stored program guidance) enter RL graphs as external inputs
(`rl/memetic_program.py:543-560`, `compile_replay` external branch); only memetic SFT links a
`memetic-guidance-producer`. The Whole-graph RL rule names forced producer turns as part of the graph; whether an
in-episode `write-text` (a `Neuralese<string>` argument materialized during the episode) should carry gradient into the
producer is the RL owner's call; guidance authored outside the episode is legitimately external.

<a id="g7-mixed-prompt-modes"></a>
### G7. The LFM3 joint-SFT inputs mix full and program-header prompts (mixture owner + supervisor)

`build_idiom_mixture.py:196` takes the program-header and whole-trajectory graph views of the renders (system turn =
program header, no tool schemas), while the teacher-twin graphs come from
`build_verified_teacher_trajectory_graphs.render_teacher_packet` with `prompt_rendering='full'` (line 1034: the
recorded system prompt and tool schemas of the teacher's runtime, 2026-10-15 to 2026-10-27). Serving uses one mode;
`checkpoint_prompt_rendering` refuses mixed declared sources without an explicit serving policy, and no LFM3 recipe
declares one. Whichever mode serves, one part of the mixture trains on a system turn the model will not see. In
`full` mode the teacher graphs also carry older runtime prompts than inference sends (the recorded prompt drifts;
program-header removes historical pieces too). Recommendation: declare program-header for LFM3 (as the LFM2.5
production line did after warm-up), pass `prompt_rendering='program-header'` through the teacher-graph builder and
re-render the teacher twins, then pin the serving mode in the recipe.

## 4. Other data defects checked

| Class | Check | Result |
|---|---|---|
| Rows with no target | per-row scan of the 34 LFM3 renders (7,381 program-header rows): `supervised_suffix_start` vs length | 0 empty, 0 with three or fewer target tokens. The objective itself does not reject all-zero rows (`own_output_supervision.py:42-43`, trainer skips them, `native_state_sft.py:440-444`) and `build_idiom_mixture.py:381-386` only counts targets; a source made only of scored kinds would pass silently (that is how the decision gap hid). Listed for the mixture owner: drop rows with zero own-output targets with a recorded reason. |
| Forced template tokens with weight | `generated_eligibility` (`own_output_supervision.py:249-274`, `dd856b7a5`) and the real-row tests | covered for the generation prompt and LFM3's opening `<think>`. Not covered: a template readout's forced `return_result` prefix (`agent.ts` template readout) if such a turn is ever rendered as a text row (none found). |
| Labels from failed trajectories | `failed_assistant_mask` (`text_supervision.py:95-140`), conversion exclusion (`native_graph_conversion.py:347-349, 405-412`), `failed_assistant_weight=0` | covered; the mixture's positive filters also require a verified root and no failed child. |
| Held/dev leakage | mixture drops train rows sharing a source group with held rows and the protected packets; dev packet provenance proves zero overlap with held v2/v3 | covered by the builder; not re-run here. |
| Duplicates dominating | exact-text duplicates per LFM3 render | one source: `natlang-native-turns-pop-static-recovery-loop-recursion-s62000-render` has 186 duplicate texts among 768 rows (one document 28 times); the mixture's near-duplicate caps bound it per tier. |
| LFM2.5 vs LFM3 rendering | `student_rerender verify` (same documents, decode, target parse, held separation) | covered per render; the refine judge row's LFM3 target (`</think>false<|im_end|>`) matches the scorer's empty-thought continuation (`serve/targets.py`). |
| Stale runtime texts | piece table (G1); `text_supervision.FAILED_TOOL_FEEDBACK` patterns still match the runtime (`native/runtime.ts:1211, 2177, 2265`); `view.py` instructions match `view.nl` | G1 fixed; no Python copies of return-error, protocol-feedback, review or `decisionPrompt` texts exist, so they cannot go stale in Python, but no parity test covers them either (decision readout: decide owner; review: G4). |
| Producer and consumer in one graph | `nativeCaptureGraphBinding` (`model-recording.ts:152-200`), `validate_model_turn_coverage` (`rl/episode_graph.py:5`) | enforced for linked turns; unlinked judge turns are the exception (G2). The 2026-10-26 truncated-producer converter family stays quarantined. |
| Rewards on an exact historical reply | `curriculum_port_record_calls` continuation tasks (`ts-host/scripts/inline-curriculum/build-port-record-calls.mjs:71-73, 93, 134, 186-187, 222`) | not in the LFM3 RL curriculum (`runs/natlang-rl-train-curriculum-lfm3-20261028/cases.jsonl`: 0 of 20,640 rows). The guard `_ill_posed_port_record_continuation` (`rl/task_source.py:16-49`) applies only to correction manifests and needs every item to be a continuation; listed for the RL curriculum owner. |
| History reasoning (`keep`) | renders bind `history_reasoning` (`render_native_turn_corpus.py:870-883`); whole-trajectory rows give earlier reasoning the 10% share | no test on a real LFM3 `keep` render (only a re-tokenized LFM2.5 row, `test_own_output_supervision_real_rows.py:178-190`); listed for the conversion owner. Teacher reasoning is preserved in history (sample row shows `<think>…</think>` in earlier turns). |

## 5. Fixes on this branch

- Standing guard: `training/model-call-kinds.json` + `tests/neuralese/test_model_call_kind_coverage.py` (call-site scan,
  route scan, per-kind SFT/RL/parity evidence, scan self-test); the 14 call sites carry `model-call-kind:` annotations.
- G1: `training/prompt-pieces/pieces.json` regenerated; `ts-host/test/prompt-pieces-export.test.mjs`.
- `scripts/neuralese_data/check_prompt_against_ts_host.py` repaired (import of a deleted module) and extended to
  count generic paragraphs in program headers and to read `.zst` renders.

## 6. Handed to owners

| Gap | Owner | What to do |
|---|---|---|
| Decision readout (cross-check) | `decide` agent | also cover `tune/rollouts.ts:145,160`, `neuralese/text-init.ts:34`, the parity fixture for `DECISION_SYSTEM_PROMPT`/`decisionPrompt`, and the teacher collector's missing scorer |
| G2 scored refinement judges | `decide` agent (mechanism), mixture owner (data) | link judge scorings as `readout: 'decision'` model turns and give them the decision action treatment; recorded objectives currently refuse any recording with a judged refinement |
| G3 answer-only drop rule | LFM3 mixture owner | exempt runtime-issued call kinds (refine-judge rows, future decision rows) from drop rule 1 with their own tier; drop zero-target rows explicitly |
| G4 careful review | supervisor | delete careful review or record it as a linked model turn before any profile enables it |
| G5 `view` server path | legacy sweep | delete `serverViewer`/`/v1/neuralese/view` or wire its turns into the recorder |
| G6 forced producers in RL | RL owner | decide whether in-episode `write-text` producers carry gradient; keep authored guidance external |
| G7 mixed prompt modes | mixture owner + supervisor | declare program-header for LFM3 and re-render the teacher twins with it |
| Port-record continuation guard | RL curriculum owner | apply `_ill_posed_port_record_continuation` to plain manifests and to mixed continue/consume items |
| `keep` history test | LFM3 conversion owner | add a real LFM3 `keep` render to the own-output real-row test |
