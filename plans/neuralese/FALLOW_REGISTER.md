# Fallow register: built capabilities not live in production

Living register required by AGENTS.md "Built capabilities must not lie fallow"
(owner, 2026-11-03). Review it between experiments. A row leaves this table when an
experiment has measured the capability and it is in use, or when a deliberate
exclusion is recorded with its reason. A blocker with no active work for 48 hours
must be resolved, replaced by a standard-practice mechanism, or excluded explicitly.
Since 2026-11-12 (AGENTS.md "How we train now") a capability enters training only as
a controlled addition with its own paired held measurement, one at a time.

`T` = `training/neuralese/natlang_neuralese`. Rows about the retired w5 machinery (interleaved controller,
JOLT+ teachers, pre-collection, rollout distillation, memetic stage and admission queues, native SFT trainer,
FP8 trainer backend, Pop dev16 watcher, DiLoCo, cloud) were removed on 2026-11-12 with that code; git history
keeps them.

## Production state (2026-11-12)

w5 is retired; the untouched base is the start policy. Training runs as bounded controlled experiments on one pinned
commit each (Experiment 0 SFT on own successes, Experiment 1 RFT, then plain GRPO), judged by paired held-out eval on
held-v2. Monitoring checks the pinned experiments and machine health; it does not re-plan.

## Paused controlled additions (APPROACH_REVIEW 2026-11-11, adopted 2026-11-12)

Switched off when the review was adopted. The "two consecutive held gains" return gate is withdrawn (owner 2026-11-17,
DECISIONS 16:45: no gate may depend on a result the current method may never produce): P2, P3, P5, P6 and P9 are being
rebuilt on the per-call stack now (`RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md`, `NEURALESE_PASSTHROUGH_20261117.md`)
and enter runs as flags, bundled by expected merit. Reviewed at every experiment boundary.

| # | Piece | Code now | Fix before it returns | Next action |
|---|---|---|---|---|
| P1 | Neuralese-declaring task variants (row N1) | live, `grpo.py --neuralese-variants` (off) | none known | first addition after two held gains (data, not machinery) |
| P2 | Whole-graph Neuralese gradients | **built 2026-11-17** (993df3f3d, NEURALESE_PASSTHROUGH_20261117.md): the pass-through, on by default in grpo and exp0_sft (`--no-neuralese-passthrough` detaches): each import's recorded producer document runs in the consumer's forward and every layer gathers its K/V/B*x, so the consumer's loss reaches the state's weights; producer table synced (cdb97e649) | none: it enters at the next pin | measure at the grpo-v4/v5 boundary: `lowering.passthrough_imports`, consumer rho |
| P3 | JOLT+ teacher attempts (privileged OPSD twin: phase 2) | rebuilt 2026-11-17 (2734e2c44): `grpo.py --teacher-attempts K` (default 0) for groups with at most `--teacher-max-successes` successes, answers excluded from the facts, trained in student form (`rl/jolt.py student_form`), students keep plain GRPO, teachers `guided_advantages` (never negative) | w5 waste fixed: K attempts inside the group with a non-zero advantage against the students; no twin on every graph. OPSD twin (`beta_s` > 0 on guided groups only) needs the learner's import path (pass-through landed 993df3f3d) | one live smoke on a DGX actor (the in-run guidance path has not run since 11-12), then on in the next bundle (`--teacher-attempts 4`) |
| P4 | Neuralese use bonus (correct episodes only) | reward code remains in `rl/rollout_math.py`; not wired into `grpo.py` | none known | after P2 |
| P5 | Memetic guidance programs | rebuilt 2026-11-17 (e23d63c21): `rl/memetic_workspace.py`, `rl/memetic_tasks.py build`, world-bridge kind `memetic`, `grpo.py --memetic-tasks --memetic-share` (default 0), anchored advantages | no real episode on the rebuilt path yet (the e2e of 11-04 ran on the deleted stage); fits not offered (no adapter actor) | build rows from the pin's pool, one real episode group on a DGX actor, then `--memetic-share 0.01` in a bundle; next: query runs as guidance-consumer data, held-program measurement |
| P6 | Self-distillation of own successes | `sft_sources.py` `rl-own` (live: sft-v3-run1); teacher-attempt successes in student form (4ee0c1760); in-loop replay `grpo.py --sd-weight` (default 0; 7321761d4) of recent rare successes | Neuralese whole-graph rows wait for the pass-through SFT rows (sibling) | in the restore smoke (`ops/restore-smoke/launch.sh`, cancelled 2026-11-17 to leave the DGX to grpo-v4; relaunch in the first DGX gap), then `--sd-weight 0.1` in grpo-v5 |
| P9 | Neuralese hindsight hints | `writeHint` returns text and runs for a teacher's root call with a verified run (6fda3e52d); index `rl/hindsight.py index` (`/home/werg/data/hindsight-index-20261117/index.jsonl`, 4,776 tasks); `grpo.py --hindsight-trajectories` | phase 1 reconstruction objective (writer trained through the hint state) needs the pass-through lowering of the hint's producer document | in the restore smoke (first DGX gap); then `--hindsight-reconstruction K` |
| P7 | Pop pre-collection, interleaved SFT windows | deleted 60d2c1209 | Pop is reserved for held evals (owner 2026-11-13) | excluded while that holds |
| P8 | QAT | off | -- | after the backbone shows held gains |

## Register

Row numbers are stable IDs. Blocker types: `wiring` (not wired into a launcher), `data` (no admitted
training input), `trainer` (no live trainer consumes it since the native SFT trainer was retired), `measurement`
(needs a paired held measurement before it is adopted).

### RL task supply (first in line: data, not machinery)

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| K1 | bgkit KR reader RL families (SearchQA, babi-in-filler, NarrativeQA, memory sessions, PubMedQA labelled) | `scripts/rl_tasks/sources/bgkit_reading.py`; 4,734 qualified train rows at `/mnt/external/natlang-rl-bgkit-kr-20261103/natlang-rl-bgkit-kr-20261103` (`BGKIT_KR_CENSUS_20261103.md`) | Qualified (reference/negative/alias replay, structure smoke, custody); in the GRPO pool (`grpo.POOLS`) since run2, not measured on its own | `measurement` | **H** (multi-call rows with Neuralese fan-in, relays and `iterateOn` reducers) | Add the source to the experiment task pool as its own controlled addition (paired held eval); keep held-v2 separation. |
| N1 | Neuralese-declaring variants of pool tasks (paired neuralese-helper framing; 1,381 rows, 112 of the ~155 measured mid-range tasks) | `scripts/rl_tasks/neuralese_variants.py`; `runs/natlang-rl-neuralese-variants-20261113` (`TASK_SUPPLY_20261113.md`) | Qualified by replay; `grpo.py --neuralese-variants` (off) | `measurement` | **H** (the plan's first Neuralese addition) | Turn on after two consecutive held gains from plain GRPO, with a paired held-v2 measurement. |
| K2 | NewsQA, MS MARCO, KILT, PubMedQA-artificial provenance stores | `/mnt/external/bgkit-data/provenance`, `browse_trees` | Unused | no passage text locally; free-form answers | L | Record an exclusion unless upstream passages are downloaded. |
| K3 | `git_history` and `qa_pairs` as git-checkable tasks | `bgkit-data/repos` (12,083 bare repos), `provenance/git_history.jsonl` | Unused; no converter | answers are free-form LLM text | M | A `git` service source whose oracle is `git` itself (commit that introduced a line, file at commit, blame). |

### Prepared training data never trained

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 22 | Broad Neuralese producer/consumer graphs and other prepared sources | `/mnt/external/natlang-fallow22-20261110/` (fetaqa s0-s5, 2wiki s0-s3, bird s0-s4 whole-chain graphs: about 13,600 train graphs, 42k imports; Luna held-shapes seeds 52116-52118; evidence-terminal D29/D33 continuations); `/mnt/external/natlang-broad-graphs-20261031` (cases, IR and turns of 8 unsharded sources); native-sft-v13-r2 recovered rows `/mnt/external/natlang-lfm3-data-20261028/native-sft-v13-r2-recovered-20261031` (stale render) | Native-graph rows; none trained by a live trainer | `trainer` | M | **Stale, not wired (2026-11-17, NEURALESE_PASSTHROUGH_20261117.md):** their system prompt is 33.4k characters vs today's 23.2k, and they are native-write spans inside producer calls, which the runtime no longer produces. Re-render from source before any use; the pass-through lowering takes such a source form. |

### Program tuning and LoRA-bank meta-learning (owner 2026-11-17; `TUNING_RESTORE_20261117.md`)

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| T1 | Per-program LoRA fits on LFM3 (program = pool family) | `scripts/neuralese_data/program_tune.py` (split/rows/plan) over `T/tune/build.py`; `serve_native_checkpoint.py --adapter-release`; `held_eval.policy_id_of`; `/home/werg/data/program-tune-20261117/` (`TUNING_RESTORE_20261117.md`) | Mechanics run on real LFM3 (split, rows, a partial fit); no task-outcome result yet | `measurement` (an actor > 15 GB for support sampling and paired dev evals) | **H** (owner 11-17) | Sample the support split (4/task), refit `ext_decisions_route`, paired dev held_eval base vs adapter; then >= 8 families. |
| T2 | Bank meta-learning over program fits | `program_tune.py bank` on `T/tune/learned/meta_bank.py` (PCA bank from the programs' own fits, Meta-SGD, held programs adapted into releases) | Tested on the tiny fixture; never run on LFM3; the 11-03 bank stopped at outer step 5 with no task outcome | `data` (needs T1 fits for >= 8 families, >= 5 support tasks with successes each) | **H** | After T1: bank over the fits, held programs' bank releases vs their own LoRA and the base on their dev splits (held_eval compare). |
| T3 | Adapter serving parity on LFM3 (GGUF LoRA on the fork and the page) | `check_tuned_lora_llama_parity.py RELEASE`, `ts-host/scripts/wllama-lora-smoke.mjs`; SGLang slots `check_sglang_adapter_parity.py` | Verified on LFM2.5 only (llama, page); SGLang random stacks only | `measurement` | M | Run both on the first real T1 release; record per-engine token/prob agreement in ENGINE_PARITY.md. |
| T4 | Factored LoRA application in native training | built 2026-11-17 (1b5b634b1): `FactoredLoRALinear`, `NATLANG_FACTORED_LORA` (default on, `0` opts out); forward-only keeps the exact substitution | LFM3 all-target (MLP included) memory unmeasured: no GPU units while SFT runs | `trainer` | S | In the next GPU window, run the all-target probe/fit on ext_decisions_route under 15 GB and record the peak. |

### Neuralese capability (returns as a controlled addition)

Design: `LEARNED_COMPACTION_OPERATORS.md`, `NEURALESE_SPAN_BASIS_AUDIT_20261103.md`.

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| C7 | Runtime compaction hooks | `ts-host/src/model/native-operator.ts` (`serverCompactor`), `native/operator-frames.ts`, `native/agent.ts` (`compaction` pluggable), server `POST /v1/neuralese/operator` (`operator/1`) | Built and tested; setting default `crisp` (text cut) | `measurement` | **H** (owner intent: Neuralese compaction) | Flip to `nl` only after a trained operator beats shuffled/zeroed/text-cut on held worlds and a ~48-sample live check. |
| C8 | Learned compaction operators | registry v3 (compact-value/params = view), `operator_rows.py` (rows + held filter + merge), `operator_ablation.py`, `native.capture` (Python, SGLang) and `grammarCapture` (fork, page; live parity at the Q8 floor), operator call + note capture in ts-host, variant `learned-compaction`; rows at `/home/werg/data/natlang-operator-rows-20261117-v1`, stage `ops/operators-v1/operator-stage.sh` | Built; step-0 ablation shows use (OPERATORS_RESTORE §7); pass-through GPU smoke passes; untrained | `pin` (stage runs at the next grpo-v4 resume with a new pin and Pop base) | **H** | `operator-stage.sh sft` from the grpo-v4 checkpoint, `held-base`, `grpo`; ablation after SFT; held on/off. |
| C2 | Directly learned soft states | `T/tune/soft.py`, `native_state_derivation.optimize_state`, `ts-host/src/neuralese/text-init.ts` | Tuning lane only | `trainer` | M | Revisit with C8: operators use text-initialized embedding rows; KV soft states stay the frozen-checkpoint cache form. |
| C6 | Native guidance writer | `applications/guidance-writer/writeGuidance.nl`, `ts-host/src/native/training-guidance.ts` | Writer answers in plain text and the host writes the native state (b069f6061). Native writing stays off: the close marker 125034 is effectively unlearned (close log-prob about -5 after 23-34 tokens, -16 to -20 after 300-token prose; produce_open rank 15k-60k) | `trainer` | M | Needs close-marker supervision on long producer spans in whatever trainer C8 adopts; switch to native writes at >=90% closed on ~48 live samples. |
| S2 | Inline-instruction edges labelled as communication | `ts-host/src/compiler/neuralese-conversion.ts` (`source_kind: inline-instruction`) | ~99k of 117,789 big2 edges carry one identical 23-token instruction payload | `data` | M | Before Neuralese data returns: give these edges an explicit `instruction` kind outside communication accounting, and cap readers per identical payload. |

### Serving

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 20 | GGUF serving of current LFM3 (local serving) | `natlang setup --native lfm-latest --gguf q8_0` | `nativeSources.lfm-latest` reads `W/ops/pop-behavior-eval/behavioural-parent.json`, which the retired Pop watcher no longer rewrites | `wiring` | L | Point `lfm-latest` at the policy an experiment promotes, then rerun setup through the ledger. |
| 23 | SGLang overlapped scheduler on the RL/held actors | `serve_native_checkpoint.py --sglang-overlap-schedule` (off by default since the 2026-11-03 one-step wrong-distribution defect) | Unused; measured +8-11% output tok/s (c=32/64, short turns) | `measurement`: at fork dbc51ddb72 the production config showed 0/192 turns over 5 nats with overlap on (control 0/192; 11-03 rate would give 0/192 with p~1%), but the 11-03 config (BF16 KV, full KV layout, 32 running) is not rerun, so fixed vs masked is open (GRPO_STEP_THROUGHPUT_20261117.md, overlap section) | M (~5% of a GRPO step) | In the first DGX gap between experiments (~20 min GPU): an overlap-on arm in the 11-03 config and a ~500-turn overlap-on sample in the production config (`serve/sglang_bench.py all`, rescore with learner arithmetic); if both are clean, overlap on by default at the following pin, with the learner's per-token rho statistics as the guard. |

## Why these went fallow (patterns)

1. **Qualification bars without a deadline.** Numerical-equality qualification
   (SGLang), capacity re-measurement (#4) and an FP8 EMA qualification (#19)
   were each treated as a precondition with no owner or date. The standard
   alternatives, such as IS-corrected off-policy sampling or reusing the
   measured envelope with a conservative margin, were never weighed.
2. **Settings frozen by incident, never restored.** QAT off, LR 1e-6, EMA eval
   raw and task eval 0 each entered during the 10-29 regression investigation
   or an emergency restart. They were then copied forward from config to config
   as "the actual controls" by custody-preserving tooling. No config field
   records *why* a value is set, so nothing prompts a revisit.
3. **Compounding scalars.** The update-budget multiplier (#1) silently quarters
   an LR that had already been measured as too small. The held-eval cadence
   (#7) is larger than every window, so the mechanism is on but never fires.
   Neither shows in a config review; both show in `train.jsonl`.
4. **Frozen runtimes lag main.** Fixes land on main, but the learner runs a
   runtime frozen hours earlier (#5). "Committed" was reported as done.
5. **Stage gates on completion of a 200-day stage.** Later recipe stages
   (adaptation, metalearner, RL) require `native_joint_sft_v4` completion or
   `mandatory_sft_only`. Throughput makes that unreachable, so everything
   downstream waits indefinitely.
6. **Prepared artifacts mistaken for progress.** DPO pairs, hindsight pilots,
   the v39 RL control and the cloud joint-sft lineage were each prepared to a
   receipt and then left without a launch owner.
7. **Idle resources not on the supervision checklist.** Pop and cw1/cw2 were
   idle while the DGX GPU ran evaluations alongside (or instead of) training.
