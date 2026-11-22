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

## Production state (reviewed 2026-11-20)

The start policy is still the untouched base (5ff07638): no trained checkpoint has beaten it on held-v3. sft-v5 (SFT on
verified teacher demonstrations and own successes) is training, and grpo-v5-run1 starts from it on pin f9664edfe
(DECISIONS 2026-11-20 05:00). The "one addition at a time" rule is withdrawn (owner 2026-11-17: engineering, not
science); additions are bundled by expected merit. Held results come from Pop.

## Controlled additions: state in the grpo-v4-run2 / grpo-v5-run1 bundle

Reviewed 2026-11-20 against grpo-v4-run2's `train.jsonl` (40 steps) and `ops/sft-v5/drive.sh` (grpo-v5-run1's flags).
A row stays here only while it is not running; the running ones are listed for their telemetry.

**Running (left the register):** P1 Neuralese task variants (`--neuralese-variants`, share 0.5: 221 of 1,378 run2
groups ran a variant), P2 whole-graph pass-through (default; 3-940 `passthrough_imports` a step), P3 JOLT+ teacher
attempts (`--teacher-attempts 2`, all-fail groups: 0-4 groups and 9 teacher successes over 40 steps, about 1% of
episodes), P5 memetic (`--memetic-share 0.02`; about 1 group every 2 steps, guided-minus-unguided gap -0.25 running
mean: telemetry, owner 2026-11-18), P6 in-loop SD (`--sd-weight 0.1`, 1-11 rows a step) and own successes in sft-v5,
P9 hindsight trajectories for teacher attempts (`--hindsight-trajectories`), C8 learned-compaction operator rows
(replay corpus `operators`, weight 0.05) and Neuralese graphs (replay corpus `neuralese-graph`, weight 0.05).

| # | Piece | Code now | State | Next action |
|---|---|---|---|---|
| P4 | Neuralese use bonus (correct episodes only) | `rl/task_rollouts.neuralese_use_bonus` (program-tune path); not in `grpo.py` | **Excluded 2026-11-20.** P2 already gives Neuralese state the task outcome's gradient, and P1 gives the opportunity to use it. A bonus for use itself rewards Neuralese whether or not it helps the outcome, which held-v3 does not score. | Reopen if held-v3's Neuralese-result tasks lag while variants run (held `neuralese_use_rate`). |
| P9b | Hindsight reconstruction objective (writer trained through the hint state) | not built (`--hindsight-reconstruction` does not exist) | The pass-through it waited for landed (993df3f3d). Teacher-attempt successes are ~0.2 a step, so the objective would train on very little data in the current loop. | **Excluded 2026-11-20** until teacher attempts produce >= 2 successes a step; the hint writer keeps training through P3's student-form rows. |
| P7 | Pop pre-collection, interleaved SFT windows | deleted 60d2c1209 | Pop is reserved for held evals (owner 2026-11-13) | excluded while that holds |
| P8 | QAT | off | **Excluded 2026-11-20.** Held-v3 and the actor serve BF16 weights (FP8 KV), so QAT cannot move the measured goal; it costs learner time. | Returns when quantized (GGUF Q8) serving of a promoted policy is the product target (row 20). |

## Register

Row numbers are stable IDs. Blocker types: `wiring` (not wired into a launcher), `data` (no admitted
training input), `trainer` (no live trainer consumes it since the native SFT trainer was retired), `measurement`
(needs a paired held measurement before it is adopted).

### RL task supply (first in line: data, not machinery)

K1 (bgkit KR reader families) and N1 (Neuralese task variants) left the register on 2026-11-20: both run in the GRPO pool (K1 since grpo-v4-run2, N1 as P1 above).

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| K2 | NewsQA, MS MARCO, KILT, PubMedQA-artificial provenance stores | `/mnt/external/bgkit-data/provenance`, `browse_trees` | **Excluded 2026-11-20** | no passage text locally; free-form answers (no exact oracle) | L | Reopen only if upstream passages are downloaded for another reason; the authored supply (rounds 17-23: 699 rows, 48 families in two days) covers reading breadth with exact oracles. |
| K3 | `git_history` and `qa_pairs` as git-checkable tasks | `bgkit-data/repos` (12,083 bare repos), `provenance/git_history.jsonl` | **In use (2026-11-20).** `natlang-rl-git-history-20261120` is in `grpo.POOLS`: 736 train rows, 10 families, over 106 repositories, through a read-only `git` service (`scripts/rl_tasks/sources/git_history.py`); every gold value is git output; TASK_SUPPLY_20261118.md section 19 | `git_history.jsonl` and `qa_pairs` stay unconverted: their answers are free-form LLM text with no exact oracle | M | Enters grpo-v5-run1 at its next resume on a new pin. Scale it by staging more repositories (`git_history.py stage`, then `cases(want=...)`) if the explore walk finds these families in band. |

### Prepared training data never trained

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 22 | Broad Neuralese producer/consumer graphs and other prepared sources | `/mnt/external/natlang-fallow22-20261110/` (fetaqa s0-s5, 2wiki s0-s3, bird s0-s4 whole-chain graphs: about 13,600 train graphs, 42k imports; Luna held-shapes seeds 52116-52118; evidence-terminal D29/D33 continuations); `/mnt/external/natlang-broad-graphs-20261031` (cases, IR and turns of 8 unsharded sources); native-sft-v13-r2 recovered rows `/mnt/external/natlang-lfm3-data-20261028/native-sft-v13-r2-recovered-20261031` (stale render) | Native-graph rows; none trained by a live trainer | `trainer` | M | **Stale, not wired (2026-11-17, NEURALESE_PASSTHROUGH_20261117.md):** their system prompt is 33.4k characters vs today's 23.2k, and they are native-write spans inside producer calls, which the runtime no longer produces. Re-render from source before any use; the pass-through lowering takes such a source form. **2026-11-20:** Excluded. The Neuralese rows that train now come from episodes on the live runtime (`sft-neuralese-graph-20261117-v1`, built by `sft_sources.py neuralese-graph` from own and teacher episodes) and from the operator sites (C8); training on a 33k-character prompt and native-write spans the runtime no longer produces would teach a distribution the model never meets. The sources stay on /mnt/external; reopen if the live corpus runs out (it trains 1 unit a GRPO step). |

### Program tuning and LoRA-bank meta-learning (owner 2026-11-17; `TUNING_RESTORE_20261117.md`)

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| T1 | Per-program LoRA fits on LFM3 (program = pool family) | `scripts/neuralese_data/program_tune.py` (split/rows/plan) over `T/tune/build.py`; `serve_native_checkpoint.py --adapter-release`; `held_eval.policy_id_of`; `/home/werg/data/program-tune-20261117/` (`TUNING_RESTORE_20261117.md`) | 2026-11-20: `ext_decisions_route` fit done (all targets, rank 16, 60x8 rows from 480 support rows; mean row loss 1.00 (steps 1-10) -> 0.71 (51-60)); route dev base 5ff07638 on Pop = 252/310 (81.3%, 2 samples, complete); adapter arm not measured yet (Pop windows only); fits of 8 more programs queued (rows from sft-sources-20261117/20 + progtune teacher: verdict 1,280, tally 3,175, tree 740, multihop 11,718, searchqa 12,568, neuralese_iterate 7,146, retail 902, pubmed 5,301) | `measurement`: Pop is the only free actor and grpo-v5 held evals hold it back to back (every unmeasured checkpoint, `--held-every-hours 1`); the DGX has no room for a second SGLang actor beside grpo-v5 | **H** (owner 11-17) | Scheduled 2026-11-20: `progdev-route-window-2` (driver) + `progdev-route-yield-2` (yields Pop within 10 s of a ready grpo checkpoint) run the adapter arm and `compare` in the next Pop window (`/home/werg/data/program-tune-20261117/progdev-window.sh`, `pop_free.py`); `progtune-fit-seq-4` fits the 8 programs one at a time (medium, 13 GB, 15-step segments; running since 19:56) and writes each `fit/release`. If no Pop window by grpo-v5's end, measure both route arms on a DGX actor then. |
| T2 | Bank meta-learning over program fits | `program_tune.py bank` on `T/tune/learned/meta_bank.py` (PCA bank from the programs' own fits, Meta-SGD, held programs adapted into releases) | Tested on the tiny fixture; never run on LFM3; the 11-03 bank stopped at outer step 5 with no task outcome | `data`: needs >= 8 program fits; 9 programs have splits, rows and plans (2026-11-20), 1 fit done, 8 fitting in `progtune-fit-seq-4` since 2026-11-20 19:56 (~2-3 min/step beside GRPO) | **H** | When `progtune-fit-seq-4` finishes: `program_tune.py bank --programs` (8 meta-train) `--held` (route, the measured one), the held program's bank release vs its own LoRA vs base on its dev split (same Pop window driver). |
| T3 | Adapter serving parity on LFM3 (GGUF LoRA on the fork and the page) | `check_tuned_lora_llama_parity.py RELEASE`, `ts-host/scripts/wllama-lora-smoke.mjs`; SGLang slots `check_sglang_adapter_parity.py` | Verified on LFM2.5 only (llama, page); SGLang random stacks only | `measurement` | M | Run both on the first real T1 release; record per-engine token/prob agreement in ENGINE_PARITY.md. |
| T4 | Factored LoRA application in native training | built 2026-11-17 (1b5b634b1): `FactoredLoRALinear`, `NATLANG_FACTORED_LORA` (default on, `0` opts out); promoted 2026-11-22 (`lfm3opti.md`) into the shared `lora+rows` core (`T/model/native_adapters.py::factored_linear`, `T/train/lfm3_fast.py`, `T/train/lfm3_learner.py`) for all SFT, RL, and probes | **In use (2026-11-22).** Shared `lora+rows` learner core is now the sole training path in `exp0_sft.py` (`r=128`) and `grpo.py` (`r=32`), with FP8 frozen-base GEMMs, trainable Neuralese rows, and atomic merged BF16 export. | none | S | Left the register 2026-11-22; monitor held-v3 on the first `lora+rows` SFT/RL cycles. |

### Neuralese capability (returns as a controlled addition)

Design: `LEARNED_COMPACTION_OPERATORS.md`, `NEURALESE_SPAN_BASIS_AUDIT_20261103.md`.

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| C7 | Runtime compaction hooks | `ts-host/src/model/native-operator.ts` (`serverCompactor`), `native/operator-frames.ts`, `native/agent.ts` (`compaction` pluggable), server `POST /v1/neuralese/operator` (`operator/1`) | Built and tested; setting default `crisp` (text cut) | `measurement` | **H** (owner intent: Neuralese compaction) | Flip to `nl` only after a trained operator beats shuffled/zeroed/text-cut on held worlds and a ~48-sample live check. |
| C8 | Learned compaction operators | registry v3 (compact-value/params = view), `operator_rows.py` (rows + held filter + merge), `operator_ablation.py`, `native.capture` (Python, SGLang) and `grammarCapture` (fork, page; live parity at the Q8 floor), operator call + note capture in ts-host, variant `learned-compaction`; rows at `/home/werg/data/natlang-operator-rows-20261117-v1`, stage `ops/operators-v1/operator-stage.sh` | Built; step-0 ablation shows use (OPERATORS_RESTORE §7); pass-through GPU smoke passes; untrained | `pin` (stage runs at the next grpo-v4 resume with a new pin and Pop base) | **H** | `operator-stage.sh sft` from the grpo-v4 checkpoint, `held-base`, `grpo`; ablation after SFT; held on/off. **2026-11-20:** Running: the operator rows train in every GRPO step as replay corpus `operators` (weight 0.05; grpo-v4-run2 and grpo-v5-run1), not as a separate stage. Next: `operator_ablation.py` on grpo-v5-run1's first held checkpoint (forward-only, fits the DGX headroom) to see whether the learned operators beat shuffled/zeroed/text-cut; that result decides C7. |
| C2 | Directly learned soft states | `T/tune/soft.py`, `native_state_derivation.optimize_state`, `ts-host/src/neuralese/text-init.ts` | Tuning lane only | `trainer` | M | Revisit with C8: operators use text-initialized embedding rows; KV soft states stay the frozen-checkpoint cache form. **2026-11-20:** Excluded: no trainer consumes it; operators take text-initialized embedding rows (C8), and KV soft states stay the frozen-checkpoint cache form of the tuning lane. |
| C6 | Native guidance writer | `applications/guidance-writer/writeGuidance.nl`, `ts-host/src/native/training-guidance.ts` | Writer answers in plain text and the host writes the native state (b069f6061). Native writing stays off: the close marker 125034 is effectively unlearned (close log-prob about -5 after 23-34 tokens, -16 to -20 after 300-token prose; produce_open rank 15k-60k) | `trainer` | M | Needs close-marker supervision on long producer spans in whatever trainer C8 adopts; switch to native writes at >=90% closed on ~48 live samples. **2026-11-20:** Excluded for now: the plain-text writer with a host-written native state works and trains through P3; native writing needs close-marker supervision on long producer spans, which the pass-through (P2) and operator rows (C8) now provide data for. Reopen with the C8 ablation result. |
| S2 | Inline-instruction edges labelled as communication | `ts-host/src/compiler/neuralese-conversion.ts` (`source_kind: inline-instruction`) | ~99k of 117,789 big2 edges carry one identical 23-token instruction payload | `data` | M | Before Neuralese data returns: give these edges an explicit `instruction` kind outside communication accounting, and cap readers per identical payload. **2026-11-20:** Excluded: the big2 conversion edges train nowhere (the live Neuralese corpora come from runtime episodes and operator sites, see 22); fix only if big2 data returns. |

### Serving

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 20 | GGUF serving of current LFM3 (local serving) | `natlang setup --native lfm-latest --gguf q8_0` | `nativeSources.lfm-latest` reads `W/ops/pop-behavior-eval/behavioural-parent.json`, which the retired Pop watcher no longer rewrites | `wiring` | L | Point `lfm-latest` at the policy an experiment promotes, then rerun setup through the ledger. **2026-11-20:** Fixed: `nativeSources.lfm-latest` now reads `W/ops/current-policy.json` (schema `natlang.current-policy-pointer/1`, the base 5ff07638 today, the deleted bf95 chain before); an experiment that meets its success criteria rewrites it. `natlang setup --native lfm-latest --gguf q8_0` rerun through the ledger. |
| 23 | SGLang overlapped scheduler on the RL/held actors | `grpo.py --actor-overlap` (auto: on for the GRPO training actor unless held evals measure on it); held engines keep `serve_native_checkpoint.py` default off | **In use (2026-11-20)** on the training actor; guard `actor_mismatch` / `actor_mismatch_shift` in train.jsonl | resolved: turned on with telemetry (DECISIONS 2026-11-20 16:30); the 11-03 config rerun stays unmeasured | M (+8-11% collection tok/s) | Watch `actor_mismatch_shift` on grpo-v5-run1's first resumed steps; on a flag resume with `--actor-overlap off`. Held engines stay off while held pairing pins `overlap_schedule: false`. |

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
