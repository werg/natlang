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

## Register

Row numbers are stable IDs. Blocker types: `wiring` (not wired into a launcher), `data` (no admitted
training input), `trainer` (no live trainer consumes it since the native SFT trainer was retired), `measurement`
(needs a paired held measurement before it is adopted).

### RL task supply (first in line: data, not machinery)

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| K1 | bgkit KR reader RL families (SearchQA, babi-in-filler, NarrativeQA, memory sessions, PubMedQA labelled) | `scripts/rl_tasks/sources/bgkit_reading.py`; 4,734 qualified train rows at `/mnt/external/natlang-rl-bgkit-kr-20261103/natlang-rl-bgkit-kr-20261103` (`BGKIT_KR_CENSUS_20261103.md`) | Qualified (reference/negative/alias replay, structure smoke, custody); in no selection pool | `wiring` | **H** (multi-call rows with Neuralese fan-in, relays and `iterateOn` reducers) | Add the source to the experiment task pool as its own controlled addition (paired held eval); keep held-v2 separation. |
| K2 | NewsQA, MS MARCO, KILT, PubMedQA-artificial provenance stores | `/mnt/external/bgkit-data/provenance`, `browse_trees` | Unused | no passage text locally; free-form answers | L | Record an exclusion unless upstream passages are downloaded. |
| K3 | `git_history` and `qa_pairs` as git-checkable tasks | `bgkit-data/repos` (12,083 bare repos), `provenance/git_history.jsonl` | Unused; no converter | answers are free-form LLM text | M | A `git` service source whose oracle is `git` itself (commit that introduced a line, file at commit, blame). |

### Prepared training data never trained

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 22 | Broad Neuralese producer/consumer graphs and other prepared sources | `/mnt/external/natlang-fallow22-20261110/` (fetaqa s0-s5, 2wiki s0-s3, bird s0-s4 whole-chain graphs: about 13,600 train graphs, 42k imports; Luna held-shapes seeds 52116-52118; evidence-terminal D29/D33 continuations); `/mnt/external/natlang-broad-graphs-20261031` (cases, IR and turns of 8 unsharded sources); native-sft-v13-r2 recovered rows `/mnt/external/natlang-lfm3-data-20261028/native-sft-v13-r2-recovered-20261031` (stale render) | Native-graph rows; none trained by a live trainer | `trainer` | M | When Neuralese returns as a controlled addition, lower them with `lower_sft_sources.py` to per-call rows for `exp0_sft.py` (whole-graph gradients only once a Neuralese capability exists to train). |

### Neuralese capability (returns as a controlled addition)

Design: `LEARNED_COMPACTION_OPERATORS.md`, `NEURALESE_SPAN_BASIS_AUDIT_20261103.md`.

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| C7 | Runtime compaction hooks | `ts-host/src/model/native-operator.ts` (`serverCompactor`), `native/operator-frames.ts`, `native/agent.ts` (`compaction` pluggable), server `POST /v1/neuralese/operator` (`operator/1`) | Built and tested; setting default `crisp` (text cut) | `measurement` | **H** (owner intent: Neuralese compaction) | Flip to `nl` only after a trained operator beats shuffled/zeroed/text-cut on held worlds and a ~48-sample live check. |
| C8 | Learned compaction operators (training data and trainer) | `T/model/learned_operators.py`, `T/data/operator_graph.py` (renderer, used by the serve engine), `T/data/operator_seeds.py`; cases and ablations at `/mnt/external/natlang-compaction-operators-20261103/` (`cases/`, `broad-cases/`, `ablation/`); the case converters, the graph builder and `eval/neuralese_ablation.py` were removed 2026-11-12 (git history before 60d2c1209 / this row's commit) | Untrained: the only trainer was the retired native SFT trainer; v1 slot smoke showed no row-specific information | `trainer` | **H** | Choose the training path (per-call rows with the producer span as target in `exp0_sft.py`, or a whole-graph learner) when Neuralese returns; then ablate on held operator graphs. |
| C2 | Directly learned soft states | `T/tune/soft.py`, `native_state_derivation.optimize_state`, `ts-host/src/neuralese/text-init.ts` | Tuning lane only | `trainer` | M | Revisit with C8: operators use text-initialized embedding rows; KV soft states stay the frozen-checkpoint cache form. |
| C6 | Native guidance writer | `applications/guidance-writer/writeGuidance.nl`, `ts-host/src/native/training-guidance.ts` | Writer answers in plain text and the host writes the native state (b069f6061). Native writing stays off: the close marker 125034 is effectively unlearned (close log-prob about -5 after 23-34 tokens, -16 to -20 after 300-token prose; produce_open rank 15k-60k) | `trainer` | M | Needs close-marker supervision on long producer spans in whatever trainer C8 adopts; switch to native writes at >=90% closed on ~48 live samples. |
| S2 | Inline-instruction edges labelled as communication | `ts-host/src/compiler/neuralese-conversion.ts` (`source_kind: inline-instruction`) | ~99k of 117,789 big2 edges carry one identical 23-token instruction payload | `data` | M | Before Neuralese data returns: give these edges an explicit `instruction` kind outside communication accounting, and cap readers per identical payload. |

### Serving

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| 20 | GGUF serving of current LFM3 (local serving) | `natlang setup --native lfm-latest --gguf q8_0` | `nativeSources.lfm-latest` reads `W/ops/pop-behavior-eval/behavioural-parent.json`, which the retired Pop watcher no longer rewrites | `wiring` | L | Point `lfm-latest` at the policy an experiment promotes, then rerun setup through the ledger. |

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
