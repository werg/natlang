# S7: Task-level reinforcement learning

Current design, 2026-10-12 (owner revived S7 RL). Supersedes the 2026-10-03 draft; its superseded parts (vLLM-fork
rollout serving, separate environment-wrapper programme, the generic family table, the long work-item list) are
deleted, not archived. Part of the [Neuralese programme](README.md); rollout records in
[S7_ROLLOUT_CONTRACT.md](S7_ROLLOUT_CONTRACT.md); reward source in
[DIRECT_TASK_EVAL_DECOMPOSITION.md](DIRECT_TASK_EVAL_DECOMPOSITION.md).

S7 improves whole natlang programs through fresh on-policy rollouts in the harness, rewarded by the same task-success
judgement that the held-task eval uses. It trains what SFT reaches badly: decomposition, choosing and finishing
calls, error recovery, stopping, and (in the end-to-end line) the Neuralese values one call writes for another.

## 1. Principles (kept)

- **Rewards come from checks.** The reward is the collector's exact terminal judgement of a task, not a model judge.
- **Rules are enforced by the host.** Types, capabilities, termination and budgets are hard; violating them fails the
  call. They are not reward terms.
- **One gradient mechanism.** Policy learning is advantage-weighted `logLikelihood` replay of recorded turns
  (`/v1/neuralese/grad`), the same replay the other stages use. No separate RL engine.
- **Infrastructure never teaches.** A rollout that fails for infrastructure reasons is unscored (`reward: null`),
  never 0.

## 2. Reward

`reward = 1` iff `eval.task_packet.classify_case(result, finish) == "task_success"` (collector outcome `done` and
`accepted`); `0` for `invalid_action`, `repetition`, `budget_exhaustion`, `model_incomplete`, `semantic_mismatch`;
`null` for `infrastructure_error`, `unresolved` and host wall-clock stops (`case_wall_budget`, a server-load
artefact; 2026-10-12, §8.3). One function, `rl.task_rollouts.reward_of`, so the RL reward and
the held-task eval can never disagree. Partial credit and the resource term (successful rollouts only) are later
additions, each measured before adoption.

**Task source.** Only train-split task rows (`split == "train"`; first source:
`runs/decision-rich-train-20261005-v2/cases.jsonl`, 72 rows, paired with the protected held 48). The preparer
refuses any other split. The protected held 48 (`runs/decision-rich-heldout-20261005-v1`) is evaluation only. Rows
get the held packet's derivation (`trajectory_contract.authored_root_eval` bound to the row's reference root by
SHA-256), so the model acts inside the child calls exactly as in the eval. More families come from more natlang
domains with exact checks (AGENTS.md: breadth from natlang domains), each added as train rows with the same contract.

For diagnosis or training of the **whole student policy**, `task_rollouts prepare --root-mode student`
keeps the train row and its exact oracle but removes authored-root replay. The default
`--root-mode authored` retains the original child-policy S7 design. The preparer records
the chosen mode and packet/source hashes in `preparation.json`; scoring rejects any
authored-root action in a student-root packet. Do not interpret authored-root successes
as evidence that the student can choose the root program. This was caught on Pop on
2026-10-14 when a nominal student probe showed zero-token root actions and empty raw
model calls despite valid task outcomes. That probe was stopped and discarded.

## 3. Rollouts: harness, collector, served student

- **Server.** The student is served by `natlang_neuralese.serve` from the direct artifacts the direct stages export
  (`--direct-artifact`, or `--base` for the base model), so RL trains the same weights and the same prompt rendering
  the direct line produces and the held-task eval scores. GPU for rollouts (BF16); the eval stays on its pinned CPU
  contract.
- **Driver.** `scripts/run_bonsai_queue.py` with `--chat-completions-url` runs the frozen TS collector
  (`dist/teacher/cli.js`), one single-case entry per rollout. A queue entry may set `temperature` (added
  2026-10-12; absent = the collector's greedy default), and `--chat-request-config` may carry
  `x_natlang_adapters` (validated `[{id, scale}]`) so rollouts sample an adapter policy. A group is `G` entries of one
  row with distinct seeds.
- **Glue.** `training/neuralese/natlang_neuralese/rl/task_rollouts.py`: `prepare` (train-only packet + group queue),
  `score` (classify every rollout, reward, `grpo.group_advantages`, route each group). Tests:
  `tests/neuralese/test_rl_task_rollouts.py`.
- **Policy turns for replay.** The collector result rows carry the `trajectory`; `score --emit-rollouts` converts it into
  `natlang.rollout/1` records (`turns` = messages/tools/target exactly as on the wire) for groups routed to GRPO.

## 4. Groups, curriculum and step 0

Per task, `G = 4..8` rollouts at temperature 1. Advantage `(r − mean)/(std + ε)` within the group
(`grpo.group_advantages`). Each group is routed by its spread:

| Group outcome | Route |
| --- | --- |
| mixed successes and failures | **GRPO** (policy gradient) |
| all fail | **rejection sampling / seeding**: draw more samples or teacher rollouts of the same row; successful trajectories go to SFT (RFT) |
| all succeed | **retire** from the active set (keep for regression checks) |
| all unscored | **infrastructure**: fix, never train |

**Step 0 is rejection-sampling fine-tuning (RFT).** While the policy solves almost nothing, groups have no spread and
GRPO gives no gradient. Step 0 then collects successful trajectories (own samples at higher `G`, or teacher rollouts,
importance-weighted and phased out per the 2026-10-03 owner decision) and trains them with the direct-SFT objective
through the shared trainer (whole-trajectory supervision). GRPO starts on the rows that thereby acquire spread.

**Task selection is a natlang policy.** Which rows and how many samples per row next round is a decision, so it is a
pluggable policy (`ts-host/src/runtime/pluggable.ts` pattern): one interface `selectTasks(groupStats, budget) →
{row, samples}[]`, crisp default (keep mixed groups; spend extra samples on all-fail rows with any historical success;
retire all-success rows), and a natural-language implementation that reads the per-row outcome histograms and
failure classes and states its choice. Shadow mode compares them before the NL one serves.

## 5. Algorithm and KL

- **GRPO update:** `rl/grpo.py round` (exists): behaviour and reference log-probabilities by replay, KL-shaped reward
  `r − β(log π − log π_ref)`, clipped sequence ratios, Adam on the adapter block via `/v1/neuralese/optim`. First
  form trains an adapter block on the serving engine; the stage form trains the full direct checkpoint with the
  shared trainer.
- **KL to the pre-RL checkpoint** is measured by the shared drift monitor (`train/drift_monitor.py`) with the pre-RL
  checkpoint as reference: response KL warn 0.5 runs the report-only behaviour evaluation. KL and base agreement
  never stop training or deny qualification (owner correction 2026-10-15). Actual task
  quality, held separation, numerics and runtime correctness remain checked. The optional
  in-loss β term is declared explicitly; default production adaptation does not penalize
  changing the base distribution.
- **Replay:** the stage mixes its train rows' SFT data (direct objective) at a declared weight; no general anchors.

## 6. Recipe stage (built 2026-10-13, commits c8bc0c7f, f18828c4)

`neuralese_rl` in `train/recipe.py` `HANDLERS`, module `natlang_neuralese.rl.stage`. One implementation per concept:
the stage owns only the RL data decision; the update is the shared direct trainer (`train/direct_sft.py`: loss,
optimizer, checkpoints, held CE, in-run task-success evaluation, drift monitor).

- **Inputs:** `rollouts` (scored `natlang.rollout/1`, from `task_rollouts score --emit-rollouts`; teacher rollouts
  as further files), and the continuation roles of the direct trainer. In a lineage the stage `requires` a direct
  phase and continues from its output implicitly (`DIRECT_TRAINING_KINDS`, the same rule as direct phase to phase);
  a standalone RL recipe pins `continue_from`/`continue_identity`/`continue_monitor_reference`.
- **Parameters:** every `neuralese_direct` parameter (defaults inherit the recipe's `neuralese_direct` defaults,
  `RL_DEFAULTS_FROM`) plus `rl_mode` (`rft`), `rl_tokenizer` (base snapshot), `rl_prompt_rendering`,
  `rft_max_success_rate` (0.75), `rft_held_fraction` (0.125).
- **Mode `rft`:** successful turns of tasks with 0 < success rate <= `rft_max_success_rate` (`rl.stage.rft_tasks`)
  become whole-trajectory documents rendered by the shared serving renderer (`native_gold_packet`, the checkpoint's
  rendering), reply as supervised suffix, task-level held split; receipt `rl-data-receipt.json`.
- **KL to pre-RL:** the stage runs the trainer with `--fresh-monitor-reference`, so the drift reference is captured
  from the continued (pre-RL) weights at update 0 instead of copying the predecessor's base reference.
- **Gate and report:** the direct gate (`qualified`: held CE and task-success cadence on the held packet);
  epochs budget, objective arguments and `--plan-only` as for direct phases.
- **Full-checkpoint GRPO (2026-10-15):** `rl_mode=grpo` uses `rl/full_policy.py` through a narrow
  objective hook in the shared direct trainer. Exact server sampling captures supply prompt token/block coordinates,
  sampled token IDs, actual terminal tokens and behavior probabilities. Token-level asymmetric clipped updates
  preserve per-sample signed advantages, including identical actions on differently rewarded trajectories.
  Ordinary text-window splitting does not truncate RL context or duplicate its updates. The same optimizer,
  recoverable checkpoints, numerical checks and task cadence serve both backbones.
- **Production held CE:** optional registered input role `rl_held_text_data` selects only its test rows, records
  their source hash and ordinary-CE meaning, and leaves fresh RL tasks available for training. The shared loader
  checks held/train source and text separation. Without this input the previous task-level RL held split remains.
- **Neuralese value replay:** `rl_block_bundle` contains exact content-addressed block bytes plus producer metadata.
  Consumers keep recorded observations while gradients replay their writers. Forced-only writers remain in the
  graph but are not invented stochastic actions. Guided/masked captures require their exact sampling law and are
  refused by the initial unconstrained-temperature-1 contract; legacy uncaptured trajectories remain RFT inputs.
- Tests: `test_rl_full_policy.py`, `test_rl_sampling.py`, `test_rl_graph_replay.py`, `test_rl_stage.py`.
  Local diagnostics and tiny-port checks are not production completion or measured model lift.

## 7. End-to-end / Neuralese path

Once the end-to-end stage (NEURALESE_DIRECT_V1.md, "End-to-end stage") has a qualified artifact, rollouts run with
consumers reading Neuralese values: producer calls write blocks (`DirectWriter`), consumers read them, the task
reward is the same. The policy term then reaches the writer through the consumer's log-likelihood (RL on writes);
sampled-payload log-densities are added once `neuralese_temperature > 0` is qualified. Report token-reading versus
value-reading consumers per round (the owner's Neuralese value requirement lives in that stage).

## 8. Smoke (2026-10-12) and diagnoses

Run dir `/mnt/external/natlang-development-data/runs/s7-rl-task-smoke-20261012/` (DGX). LFM2.5-350M base, BF16 on
GPU, `natlang_neuralese.serve --base`, 8 train rows (`decision_extract_chain` s7202-0..7) × G=4, temperature 1,
case budget 300 s, 32 model requests; 4 parallel queue runners; ledger unit 7 GB.

```
PYTHONPATH=training/neuralese .venv-neuralese/bin/python -m natlang_neuralese.rl.task_rollouts prepare \
  --cases runs/decision-rich-train-20261005-v2/cases.jsonl --out $D/round0 --rows 8 --group 4 --temperature 1.0 \
  --case-seconds 300 --max-model-requests 32
python3 scripts/memory_ledger.py run --unit s7rl-round0 --budget-gb 7 --reserve-gb 2 --env PORT=8098 \
  --env RENDER=full -- scripts/s7_task_round.sh $D/round0 4     # serve + queue + score
```

### 8.1 Results

| Run | Rendering | Rollouts | Outcomes | Successes | Groups with spread |
| --- | --- | --- | --- | --- | --- |
| round0 | full | 32 | 30 semantic_mismatch, 2 invalid_action, 0 infrastructure | 0 | 0 / 8 |
| probe (row 0) | full | 4 | 4 semantic_mismatch | 0 | 0 / 1 |
| probe (row 0) | program-header | 4 | 4 model_incomplete | 0 | 0 / 1 |

Wall time 15–88 s per rollout (mean 41 s) on GPU; the whole round about 6 min. Every group routes to rejection
sampling: GRPO has no gradient on the base, so no `grpo.py round` was run (it would be a no-op: all advantages 0).

### 8.2 Diagnosis of the base failures (held-task 0/48)

- **The held eval's 27 `model_incomplete` are caused by scoring the base under `program-header` rendering.**
  `apply_prompt_rendering(..., "program-header")` drops the runtime instructions *and passes no tool schemas*
  (`tools=None`); a base model that never trained on that rendering has no tools to call. Reproduced on train row 0:
  program-header gives 4/4 `model_incomplete` (19 completion tokens per rollout, root eval quiesced); the same row
  under `full` rendering gives well-formed `return_result` tool calls in every rollout. The base reference for the
  held eval must be scored under `full` rendering (its native contract); program-header is valid only for
  checkpoints trained on it. Routed to the held-eval owner (Pop LFM line) via coord.
- **Under full rendering the base fails for capability, not plumbing.** 30/32 child calls answer with one immediate
  `return_result` holding a hallucinated `Evidence` (quote = the question, role = "What"); 0/32 call
  `packet.read()`; 2/32 try an `eval` and fail. Ruled out: rendering/template (tool calls parse, stop reasons
  normal), infrastructure (0 unscored), sampling (temperature 1, distinct seeds, 4 distinct answers per group),
  scorer (the same classifier and packet derivation the held eval uses on teacher reference results).
- **The 10-09 smoke's six unscored "driver" families were infrastructure:** 59 of 72 decision rollouts failed with
  `ENOENT uv_cwd` / missing `prelude.js` (the driver ran from a temporary clean checkout under
  `~/.claude/jobs/.../rl-clean` that was deleted mid-run) and 18 with `ECONNREFUSED :8097` / `terminated` (the
  reference server stopped). They were correctly unscored; no model or scorer fault.

Consequence: RL on this row family needs a policy with nonzero success first — step 0 (RFT on teacher or direct-SFT
successes), or start S7 from a direct-SFT LFM checkpoint (the stage design already says RL starts from the direct
artifact, not the base).

### 8.3 Direct-SFT student (Pop LFM Muon main, step 10006) and the first GRPO update

Same 8 rows × G=4, temperature 1, served from `runs/pop-lfm-muon-main-v2-export-step10006` with its own
`program-header` rendering. The export's source proof pins Pop paths (`/srv/storage/...` HF cache and launch
receipt), so `s7_task_round.sh` serves it in a container (`SERVE_IMAGE=natlang-neuralese:26.09`) that mounts the
local HF cache and the export's byte-identical `artifact.launch-receipt.json` at those paths. The loader still checks
every hash. (The step-9000 export had no receipt beside it; luna v8 lacked its anchor beside it.)

```
python3 scripts/memory_ledger.py run --unit s7rl-round0-muon10006 --container s7rl-round0-muon10006 --budget-gb 8 \
  --reserve-gb 2 --env PORT=8098 --env DIRECT=$E --env SERVE_IMAGE=natlang-neuralese:26.09 \
  --env RECEIPT=$E/artifact.launch-receipt.json --env CONTAINER=s7rl-round0-muon10006 \
  -- scripts/s7_task_round.sh $D/round0-muon10006 4
# one GRPO round: emit natlang.rollout/1, zero xs adapter (rank 8), grpo.py round (lr 1e-3, beta 0.02, clip 0.2)
... --env UPDATE_FROM=$D/round0-muon10006 -- scripts/s7_task_round.sh $D/update-muon10006
# re-score with the updated adapter (collector passes x_natlang_adapters via --chat-request-config)
... --env RESTORE_ADAPTER=$D/update-muon10006/policy-r1/adapter.nzb -- scripts/s7_task_round.sh $D/round1-muon10006 4
```

| Run | Rollouts | Outcomes | Successes | Groups: grpo / retire / rejection |
| --- | --- | --- | --- | --- |
| round0, Muon 10006 | 32 | 11 task_success, 11 invalid_action, 7 semantic_mismatch, 3 budget_exhaustion | 11 (34 %) | 1 / 2 / 5 |
| round1, + GRPO adapter | 32 | 8 task_success, 15 budget_exhaustion (all host wall budget), 5 semantic_mismatch, 4 invalid_action | 8 | 2 / 1 / 5 (old routing) |

- Unlike the base, the direct student solves rows end to end: two rows succeed 4/4 (retire), row s7202-1 is mixed
  (3/4: the GRPO group), five rows fail in every sample. Failure classes (non-successes): `answer` rejected 7,
  `invalid-binding` 4, `missing-fresh-result` 4, `undeclared-field` 2, `typescript-syntax` 1, no result row 2,
  aborted 1. Wall 79–314 s per rollout (mean 144 s; the student uses more turns than the base).
- **GRPO round (`natlang.grpo-round/1`):** 32 rollouts, 30 usable (2 without a result row have no turns), 0 unscored,
  1 group with spread, reward mean 0.37, 22 logLikelihood terms, ratio 1.0 (on-policy, zero adapter), 0 clipped, KL
  estimate 0, surrogate −1.48, 32 s. Adapter `nz1_m6a6…` in `update-muon10006/policy-r1/`. The pipeline
  rollout → classify_case reward → group advantage → replay gradient → adapter → collector sampling with the adapter
  runs end to end.
- **Round1 is not a policy measurement.** Budget exhaustion rose 3 → 15, all `case_wall_budget` (300 s host
  wall stop), while the model produced *fewer* replies (214 vs 361) and far fewer completion tokens (7.0k vs 16.8k):
  decode throughput dropped about 4× per token. Round1 ran with the Mellum v4 relaunch training on the same GPU (round0
  did not), and with the adapter applied per request; either slows serving. Not yet separated: the next round runs a
  matched no-adapter control in the same unit and seeds. Among rollouts that finished, round1 had 8/17 successes,
  round0 11/29; one tiny step on one group is not expected to move this measurably.
- **Matched A/B (2026-10-13), resolving the above.** Same 8 rows, same seeds (seed base 11000), 4 samples, wall limit
  raised to 900 s, both arms concurrently on one server (4 queue runners each) so they share GPU contention; unit
  `--no-resumable` (two earlier attempts were preempted by Mellum v4 relaunches).

  | Arm | Successes | Outcomes | Successes per row | Mean / max wall | Replies | Completion tokens |
  | --- | --- | --- | --- | --- | --- | --- |
  | adapter (GRPO r1) | 11/32 | 11 success, 9 invalid_action, 4 semantic_mismatch, 8 model-request budget | 0 4 0 4 0 3 0 0 | 213 / 797 s | 407 | 51.0k |
  | control (no adapter) | 12/32 | 12 success, 12 invalid_action, 8 model-request budget | 0 4 0 4 0 4 0 0 | 208 / 897 s | 406 | 44.6k |

  No policy difference (one step on one group, as expected) and no serving difference: no wall-clock stops in either
  arm, equal wall times. Decode probe (`scripts/s7_decode_probe.py`, one recorded 418-token request, greedy,
  alternating ×3): idle server 16.3 tok/s base vs 14.7 with adapter (1.1×); under the A/B load 31.6 vs 41.0 (adapter
  faster; noise at 23 generated tokens per request). The adapter path does not cost 4×; round1's slowdown was GPU
  contention from the concurrent Mellum training. The 8 budget stops per arm are the 32-model-request limit (reward 0),
  all on rows that fail in every sample. The same 3 rows succeed in both arms; the other 5 never do, so GRPO on this
  set has almost no signal: the next step is more rows and RFT on the all-fail rows, not more GRPO steps here.
- **Reward fix from this:** a `case_wall_budget` stop is a collection resource limit that depends on server load
  (the queue journal itself labels it "not a language deadline or model-negative label"), so `reward_of` now makes
  it unscored (null), like infrastructure; `model_request_budget` stays reward 0. Rescored: round0 29 scored, 11
  successes, 1 group with spread; round1 17 scored, 8 successes, 0 groups with spread. The held eval keeps counting
  wall stops as failures (its CPU contract is fixed and serialized).
- A ledger preemption by the Mellum v4 relaunch (high priority) killed the first update attempt mid-`grad`; the
  requeued unit completed. Resumable smoke units are requeued after preemption and also after failures: cancel
  stale ones (`memory_ledger.py cancel`).

## 9. Next work

1. Scale the GRPO smoke: more rows (all 72 train rows), G=8 so more groups have spread, several rounds with
   re-scoring; RFT step 0 on the five all-fail rows (teacher rollouts) and on the student's own successes.
2. Evaluate the RL adapter on the held 48 (task-success cadence) against the step-10006 export, matched rendering.
3. Train the full direct checkpoint instead of an adapter: `neuralese_rl` handler + recipe (`extends` the direct
   recipe), drift monitor against the pre-RL checkpoint; task-selection pluggable (crisp default, NL shadow).
4. Held eval re-scores the base under `full` rendering (Pop; requested 2026-10-12).
5. End-to-end rollouts with Neuralese consumers.
