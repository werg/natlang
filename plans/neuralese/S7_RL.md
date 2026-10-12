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
`null` for `infrastructure_error` and `unresolved`. One function, `rl.task_rollouts.reward_of`, so the RL reward and
the held-task eval can never disagree. Partial credit and the resource term (successful rollouts only) are later
additions, each measured before adoption.

**Task source.** Only train-split task rows (`split == "train"`; first source:
`runs/decision-rich-train-20261005-v2/cases.jsonl`, 72 rows, paired with the protected held 48). The preparer
refuses any other split. The protected held 48 (`runs/decision-rich-heldout-20261005-v1`) is evaluation only. Rows
get the held packet's derivation (`trajectory_contract.authored_root_eval` bound to the row's reference root by
SHA-256), so the model acts inside the child calls exactly as in the eval. More families come from more natlang
domains with exact checks (AGENTS.md: breadth from natlang domains), each added as train rows with the same contract.

## 3. Rollouts: harness, collector, served student

- **Server.** The student is served by `natlang_neuralese.serve` from the direct artifacts the direct stages export
  (`--direct-artifact`, or `--base` for the base model), so RL trains the same weights and the same prompt rendering
  the direct line produces and the held-task eval scores. GPU for rollouts (BF16); the eval stays on its pinned CPU
  contract.
- **Driver.** `scripts/run_bonsai_queue.py` with `--chat-completions-url` runs the frozen TS collector
  (`dist/teacher/cli.js`), one single-case entry per rollout. A queue entry may set `temperature` (added
  2026-10-12; absent = the collector's greedy default). A group is `G` entries of one row with distinct seeds.
- **Glue.** `training/neuralese/natlang_neuralese/rl/task_rollouts.py`: `prepare` (train-only packet + group queue),
  `score` (classify every rollout, reward, `grpo.group_advantages`, route each group). Tests:
  `tests/neuralese/test_rl_task_rollouts.py`.
- **Policy turns for replay.** The collector result rows carry the `trajectory`; `score` is to convert it into
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
  checkpoint as reference: response KL warn 0.5 runs the behaviour gate, stop at 1.5, gate pass ≥ base − 0.1 (same
  thresholds as the direct stages, LINE_PARITY.md). The in-loss β term is per-rollout and secondary.
- **Replay:** the stage mixes its train rows' SFT data (direct objective) at a declared weight; no general anchors.

## 6. Recipe stage

A `neuralese_rl` handler in `train/recipe.py` `HANDLERS`, module `natlang_neuralese.rl.stage` (to build), recipes
`extends` the lineage's direct recipe and override only RL parameters: `rows` (task-row binding), `group`,
`temperature`, `rounds`, `rft_min_successes`, `beta`, `clip`, `lr`, drift-monitor parameters. Inputs: the direct
checkpoint (`continue_from`), task rows, monitor reference. Each round: export → serve (ledger GPU unit) → queue →
score → RFT or GRPO step → drift check → held-task eval subset on cadence. Result: `report.json` plus the best
checkpoint retained by `task_packet.retain_best_checkpoint` on held-task successes.

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

## 9. Next work

1. Re-run round0 from the current direct-SFT LFM checkpoint (`--direct-artifact`, matching its rendering) to find
   groups with spread; held eval re-scores the base under `full` rendering (Pop).
2. `score --emit-rollouts`: `natlang.rollout/1` records from collector results, then one `grpo.py round` on a group
   with spread; adapters through the collector (chat controls do not yet carry `x_natlang_adapters`).
3. RFT step 0 on successes (teacher rollouts of train rows where the student has none).
4. `neuralese_rl` handler + recipe; task-selection pluggable (crisp default, NL shadow).
5. End-to-end rollouts with Neuralese consumers.
