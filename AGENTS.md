# How we train now: bounded controlled experiments (owner, 2026-11-12)

`plans/neuralese/APPROACH_REVIEW_20261111.md` is the plan; these are the standing rules for model training. No w5
checkpoint beats the untouched base on held-out (base 5ff07638 54/118, step 92 50/127, steps 110-134 down to 39/128),
so w5 is retired (data kept, never resumed) and the base is the start policy.

- **Precision.** Fine-tuning keeps FP32 master weights and FP32 optimizer state and computes in BF16
  (`scripts/neuralese_data/exp0_sft.py`). Pure-BF16 parameters with stochastic-rounding writes are not used for
  LFM3: at our learning rates those updates were rounding noise (review section 1.1).
- **Bounded controlled experiments decide.** Each experiment is one collect -> train -> eval cycle on one pinned
  commit (actor, harness, scorer, learner and evaluator all run it; record the SHA), changes exactly one thing
  against its baseline, declares its success criterion before it runs, and is judged by paired held-out eval.
  Experiments are small (hours). Record each with its result in `plans/neuralese/DECISIONS.md`.
- **Improvements enter at the next experiment, never mid-measurement.** Code landing on main does not reach a
  running experiment: a pinned run uses its own copy of its commit (a detached checkout or a pinned copy of
  `ts-host`; no commits are made there, and it is removed afterwards). A measured win becomes the baseline of the
  following experiments, with an opt-out switch. A correctness bug found in a running experiment invalidates it:
  stop it, fix it, rerun it.
- **Learner data path: per-call sequences.** Each model call of an episode is one (prompt, completion) sequence. The
  rendered prompt is zero-weight context, the sampled completion is the target, and the episode reward is the
  weight. Experiment 0 is SFT on own successes, Experiment 1 is a rejection-sampling fine-tune (256 tasks x 8; pass
  is >= +8 held successes, p < 0.05), then plain GRPO (group 8, per-call sequences, KL to the start policy).
  Whole-graph Neuralese gradients return as a controlled addition once there is Neuralese capability to train (use
  rate today 5-13%).
- **Off until each returns as its own controlled addition with a paired measurement:** the JOLT+ teacher and
  privileged twin, the Neuralese bonus, memetic guidance programs, expert-iteration window shares, Pop
  pre-collection, interleaved SFT windows and QAT. They come back one at a time, after two consecutive held gains
  from plain updates. Neuralese-declaring task variants come first, because they are data, not machinery.
- **Evaluation gets real compute.** Paired held-out eval on a large pinned set decides every comparison (held-v2
  below; McNemar exact on matched task-samples, task sign test, loop rate beside it). dev16 is a smoke test at most,
  never a gate.
- **Heartbeats check, they do not re-plan.** A monitoring cycle checks that the pinned experiments run correctly and
  the machines are healthy, and fixes breakage. Plans change between experiments, from their results.
- **The training and RL stack uses standard tooling** (an HF training loop, the SGLang server) and minimal crisp
  code. Build infrastructure only when the next experiment needs it; no new transitions, qualification scopes,
  controllers or custody schemas.
- Agents adopt plans like this one themselves: waiting for the owner's OK on a technical course change is a defect
  (owner, 2026-11-12).

## GRPO loop: plain GRPO from the base, overlapped (engineering, 2026-11-12)

`scripts/neuralese_data/grpo.py train` is the review's "Hour 24-48" GRPO (a few hundred lines; no controller, custody or
transitions). One ledger unit holds the learner and, as a child process, its training actor (held-v2's pinned engine:
SGLang vendor projection, FP8 KV; a separately admitted actor unit deadlocked behind the learner's claim). Collection never stops: `--slots` work-stealing episodes over an endless
shuffled stream of task groups (8 samples; pools: the w5 pool plus the verified bgkit KR rows, fallow K1; held-v2 excluded
with its split groups and every row sharing a source group; `WORK/pool.json` records the pool shas). The learner trains on
whatever groups finished since its last step (16-128; stragglers join a later step, groups more than 2 versions stale are
dropped). Tasks (EPISODE_COST_20261112): 85% from tasks whose measured base pass rate is in [0.1, 0.9] (exp1-rft
`base-pass-rates.json`, re-read every 10 min; a task's own latest scored GRPO group replaces its rate, so tasks the policy
solves 0/8 or 8/8 leave the band and explored ones enter it; exp1's pass-rate collection stopped 2026-11-13), 15% unmeasured pool tasks; `ext_codebase-{py,natlang}_codebase_*` excluded
(repos missing on the DGX). Episode budget 40 model calls (a budget stop scores 0; the collector has no generated-token
budget yet), wall clock 3600 s as a safety net only (unscored). Reward = task outcome only. Per-call sequences with the actor's exact prompt and sampled ids
(`lower_rollout_turns.episode_turns`, chained as in `exp0_sft.chains_of`); sampled decision calls train as one action each
(`decision_turns`: log softmax of the learner's option scores over prompt + each captured continuation); Neuralese-state
consumer calls train too, the imported state fixed context spliced from the actor's block store as the fork fed it (K/V
pre-RoPE, conv B*x; `lfm3_fast_train.import_tensors`, no producer gradient; `plans/MODEL_FACING_CHANGES.md` 2026-11-13); loss on sampled tokens
`-min(rho, 2) * A * logp + 0.02 * k3(start || policy)`, `A` group-normalized; FP32-master AdamW, LR 3e-6 (check 4), BF16 compute
(`exp0_sft.load_model`, `lfm3_fast_train`). Weight sync each step: BF16 weights -> pinned buffer -> actor's HF safetensors ->
SGLang `pause_generation(retract)` / `update_weights_from_disk` / `continue_generation` in a background thread; a pruned
native checkpoint every 2 steps and the FP32 state (masters + moments, `/mnt/external/natlang-grpo-v1`) every 4 (grpo-v1 runs `--state-every 1`: a guard stop loses no step), also in
background threads. The whole run executes from its pinned copy (`scripts/pin_checkout.py create COMMIT
WORK/pin-<c12>`: detached worktree, ts-host rebuilt there; launch that copy's grpo.py with it as workdir; grpo.py refuses a
live checkout unless `--unpinned`), and every child (actor, episodes, held_eval measure/compare) gets its PYTHONPATH, so
held results record the pin's commit and series; adopting newer code is a new pin and `--resume` (when the series changes, grpo.py moves `held/step-00000` to `step-00000-series-<old>`, measures the start policy in place on the actor, restores its weights and measures the current step at once). After each step it deletes actor-store state blocks untouched since 10 min before the oldest outstanding group was created (`collect_blocks`; the store touches a block on every read and repeated write; `--keep-rollouts` keeps all). Held evals run on the training actor itself (it serves the
current policy with the held-v2 engine pin): new collection pauses, `held_eval measure` (held-v2; 4 samples for the start
policy before the first step, then 2 samples every 4 h on that step's checkpoint) and `held_eval compare` with step 0
(`WORK/held/step-N/`); `--held-extra CKPT,...` measures other native checkpoints the same way after step 0 (loaded into the
actor in place, `held/extra-<sha12>/` with a compare against step 0, 2 samples by default), then restores the start policy. `--held-remote pop` (owner throughput decision 2026-11-13; run2 uses it) moves the
held evals to Pop's GPU and the DGX never pauses: the start policy once as `held/step-00000-pop` on Pop's engine (compare
pairs only one engine), then the newest native checkpoint every `--held-every-hours` or back to back if an eval takes
longer (`pop_rl_actor.py serve-checkpoint`: rsync of the pruned checkpoint, the actor runs a `git archive` of the run's
commit on Pop, Pop keeps 2 checkpoints), measured by this run's pinned harness from the DGX against Pop's endpoint
(`--held-remote-slots` 20, `held-pop.log`). Step sizing (smoke): ~27k forward tokens per episode (chaining already merges 282 of 329
consecutive calls; 38 diverge inside the sampled reply, 9 in the prompt), learner 600-800 tok/s while sharing the GPU with
decode at `--activation-gb 4`; defaults now 12-24 groups per step, `--activation-gb 12`, 8 collector slots during the learner phase
(`--learner-phase-slots`: check 4 measured the learner 3x faster with actor decode at 32 slots than at 48-64; the run logs
actor tok/s per phase). The KL
reference runs the policy's own training graph (a no_grad pass compiles other kernels: KL 5e-4 at identical weights). After each in-place update the front attests what it serves (`POST /v1/neuralese/rebind-policy`:
the step's checkpoint sha, else the served weights' sha; the native engine keeps the start checkpoint). `train.jsonl` logs per step: reward
mean, spread fraction, KL, rho (mean, pre/post-cap max, capped fraction, |log rho|), learner and actor tokens/s, phase seconds.
Memory (smoke 2026-11-12): on the GB10 the unit's MemoryMax (= ledger budget) also bounds its CUDA allocations, so the
budget must cover GPU + host: learner CUDA 19.3 GB before AdamW state, 32 GB peak at step 1, about 36 GB with the FP32 moments
(FP32 masters + moments + FP32 gradients alone are 27 GB), host about 5 GB plus the collector node processes (in the same
cgroup, 0.3-0.7 GB per slot; shared collectors save about 0.25 GB a slot), plus the child actor (kv-gb + 24). The actor's
weight file lives on /dev/shm (the shared NVMe wrote at 60-230 MB/s under load); pruned checkpoints go to /mnt/external
(last 2 plus held-evaluated ones), the single FP32 training state (20.7 GB, temp file renamed over the previous) to NVMe.

## Held-v2: the experiments' held-out set (engineering, 2026-11-12)

`/home/werg/data/lfm3-rl-dgx-20261029/ops/held-v2/manifest.json` (sha dfe0a915): 256 pool tasks (the w5 pool b3fef4c7), 91
families, levels 63/64/64/65, 85 effect-state, keeping w5's 64 held tasks; disjoint from Experiment 1's selection, Experiment
0's training tasks, w1-w5 attempts and their `split_group` siblings. Training code excludes it with
`held_eval.exclusion_ids(manifest)` (held ids plus siblings, 1,170 ids; also `held-v2/exclusion-ids.json`). Never train on
these ids. Built by `python -m natlang_neuralese.rl.held_eval pin` (`--include-manifest`, `--avoid-ids`, `--avoid-from`).
Measure with `held_eval measure --manifest M --out DIR (--endpoint URL[=SLOTS] | --checkpoint CKPT) [--samples 4]`
(work-stealing episodes, resumable, sample-major so a cut run stays balanced; `serve` launches/reuses a ledger SGLang actor) and
compare with `held_eval compare A/result.json B/result.json` (McNemar on matched samples, task sign test, task-bootstrap CI,
loop rate). A wall-clock timeout is unscored and reported as `timeouts`, never a failure; compare only results with one code
commit and series (`same_code`, `same_series`). Baselines and the A/A noise floor: `ops/held-v2/runs/`. The eval engine is
pinned in `held-v2/engine.json` (SGLang, vendor projection, FP8 KV, `--sglang-kv-gb 16`, 128 running; one engine per
checkpoint): `measure` refuses another actor (`--allow-other-engine`), `serve` launches exactly it, and `compare` refuses to
pair results with different attested engines (`serving_engine`, export format). Interim A/A (base twice, independent seeds,
shared actor, 1 sample on 56 matched tasks): 10 discordant pairs (18%), split 8:2, delta -10.7 pts, task-bootstrap CI
[-21, 0], p=0.11: one-sample differences under ~8 pts at 128 pairs are noise; read held-v2 at 4 samples (~3 pts at 1,024
pairs). Episodes take ~15 min at the shared actor's load and ~250 MB host memory each (budget the ledger unit for slots).

## GB10 CUDA contexts fail under memory fragmentation (engineering, 2026-11-12)

`CUDA_ERROR_OUT_OF_MEMORY` from `cuDevicePrimaryCtxRetain` with tens of GB free (kernel: `NVRM ... NV_ERR_NO_MEMORY ...
kgrctxAllocMainCtxBuffer`) is fragmentation: a context's buffers need physically contiguous pages, and the Normal zone had
almost no free order >= 8 blocks (`/proc/buddyinfo`). Measured: context creation failed whenever order-8 blocks were ~3,
succeeded at 12+, and 30/30 with the compactor running. `natlang-gb10-compactor` (persistent user service,
`scripts/systemd/`) faults THP-advised memory whenever under 512 MB is free in such blocks (direct compaction, no root);
keep it running. `common/cuda_context.py`: `ensure_context()` (serve_native_checkpoint, grpo, exp0_sft) and
`start_sglang` retry a context failure 3x after a compaction nudge and 20 s. Root fix (owner): `vm.compaction_proactiveness`.

## RL actor on the DGX: one engine, many slots, absolute KV size (engineering, 2026-11-12)

Collection decode on the GB10 is memory-bandwidth bound: each step reads the weights (~3.5 GB) plus every running
sequence's full-attention KV (12 KB per context token BF16, 6 KB FP8; RL prompts average 11.5k tokens). Exp0/w5 actors
ran 16-38 slots, 30 running sequences, pool 93% full with queued requests, 8-14 tok/s per stream. Rules:
- Run ONE SGLang engine per GPU and give it all slots (exp1/held/GRPO share it): two engines time-slice the GB10 and
  every kernel waits for the other's slice. Serve with `serve_native_checkpoint.py --engine sglang` defaults:
  128 running, `--sglang-kv-gb 12` (pool in GB; tokens and SGLang's mem fraction are derived from the memory free at
  launch, so other jobs neither shrink the pool nor get grabbed). Ledger budget = `--sglang-kv-gb` + 24 GB (the front reloads the native model for write_text).
  Do not pass `--sglang-mem-fraction` (SGLang reserves (1 - fraction) of free memory: 0.25 left a few GB of KV).
- Collector slots >= 64 per DGX actor; measure with `held_eval measure`. A slot costs ~250 MB of host memory in its own
  node + python processes; episodes share node processes, 8 each, by default (`cli.js --serve`; `NATLANG_COLLECTOR_SHARED=0` opts out).
  Details and numbers: `plans/neuralese/ACTOR_THROUGHPUT_20261112.md`.

## HF-path fine-tuning speed: elementwise and dense sliding attention, not matmul (engineering, 2026-11-12)

`exp0_sft.py train` ran 1.8-1.9k fwd+bwd tokens/s on an idle GB10 (11k-token chains). Profile: matmuls (nvjet) 1.4 of 5.1 GPU s at about 90 TFLOPs; unfused
elementwise about 2.5 s; sliding layers as a dense masked 11k x 11k mem-efficient kernel 0.9 s (window 256); FP32-master accumulate + AdamW + copy-back 2%.
`scripts/neuralese_data/lfm3_fast_train.py` (reads the HF instance; vendored sources stay hash-pinned): `prepare(model)`, `hidden(model, ids, lengths=None)`,
`linear_nll(h, weight, targets)`, `token_logprobs(model, ids, positions, targets, lengths=None)` (no grad: GRPO behaviour/reference pass). The decoder layer is two
functions over explicit tensors (conv, attention), each `torch.compile(fullgraph, dynamic)` once and shared by all layers of its kind; attention is varlen flash
(`torch.nn.attention.varlen`, sliding layers window (255, 0), exact); `lengths` packs documents into one row (positions restart, attention and conv taps stay
in the document); the conv is seven fp32 shifted adds; `linear_nll` keeps only the fp32 log-sum-exp and recomputes each 4096-row logits chunk once in backward
(the stock checkpointed CE ran the head 4x and stored fp32 logits). Rounding points follow HF. `exp0_sft`: `token_nll` uses `linear_nll` on prepared models;
`set_checkpointing` recomputes layers beyond `--activation-gb` at 88 KB saved per token-layer (measured 84.8); a step's chains are packed into rows up to
`resident_tokens` (the longest row the budget holds unrecomputed, about 10k at 20 GB; `--pack-tokens`, 0 off): SFT v2 chains are short (median 2.0k).
Measured (2026-11-12, GPU shared with two SGLang eval actors at 94-96% util, so absolute rates are about a third of idle; A/B alternated back to back):
fwd+bwd all layers recomputed 1370 +- 39 (ed2ac83c0) -> 1471 +- 26 (new layers) -> 1496 +- 8 tok/s (+ fused CE), GPU s/chain 9.04 -> 8.36 -> 7.90;
packing on 20 SFT v2 chains 1739 +- 29 -> 1773 +- 20 (4k rows) -> 1803 +- 22 (8k); no-grad logprob pass 2982 +- 366 (stock HF) -> 7330 +- 1270.
Agreement: new vs ed2ac83c0 gradient cosine 0.998 / rel-L2 0.063, loss within 6e-4; per-token logprob vs stock BF16 max 0.61 / mean 0.0195 (the BF16 floor),
packed rows the same (0.69 / 0.0194). Against FP32 (`--fp32-reference` semantics, final code, 6 chains of 5-5.5k tokens): fast cosine 0.9963 / rel-L2
0.089, stock 0.9964 / 0.086, fast vs stock 0.9978 / 0.067, loss max |diff| vs FP32 0.013 / 0.009 (another 6 chains: 0.9978 / 0.066 vs 0.9973 / 0.075):
the fast path is as far from FP32 as stock HF, i.e. BF16 noise. Tried and dropped: selective recompute
of the SwiGLU product (saved activations -18%, but 1949 +- 55 vs 1987 +- 38 tok/s at equal budget).
Kernel fusion is at its limit on this hardware (layer bench, real dims, T=6000): Liger-Kernel (HF-exact `llama` RMSNorm, fused add+RMSNorm, SwiGLU;
it does not trace inside `torch.compile`) runs at the speed of inductor's kernels (non-matmul GPU ms per conv layer 12.13 vs 12.18; SwiGLU backward
2.04 vs 1.97 ms, about 234 GB/s, 85% of the GB10's 273 GB/s), so it is not used. Removing those bytes needs norm/SwiGLU fusion into the GEMMs, but
inductor's Triton GEMM templates (max-autotune, forced past its SM-count gate) run at 34-35% of cuBLAS nvjet on sm_121 and cuBLASLt has no such
epilogues; the matmul-only floor bounds what any fusion could still win at about 8% (conv layer 33.7 vs 39.5 ms, attention 43.7 vs 45.3).
BF16 matmuls at about 86-90 TFLOPs cap fwd+bwd near 9.8k tok/s with zero other work, so 10k tok/s is out of reach for 1.7B on one GB10.
GPU kernel time equals wall time per chain (no launch gaps), so CUDA graphs buy nothing at these lengths; the old per-instance layer compile already fused
(3 shared graphs, no breaks), so the gain is the sliding attention (mem-efficient masked kernel + GQA copies, about 0.47 GPU s/chain) and the CE (0.45).
Matmuls are now about 60-65% of GPU time; what remains is elementwise in the compiled layers (about 22%), flash (8%) and layer recompute.
`--stock-model` opts out. Check: `exp0_sft.py bench --fp32-reference`, `tests/neuralese/test_lfm3_fast_train.py`. The native learner
(`model/hf_port.py`) is a separate path (APPROACH_REVIEW section 1.2).

## Retired w5 machinery (removed 2026-11-12; git history keeps it)

The w5 RL controller and the native BF16 trainer stack are gone: interleaved RL with SFT windows (`rl/interleave.py`),
per-group JOLT+ teacher planning (`rl/teacher_feed.py`), Pop pre-collection of the next round (`rl/precollect.py`),
expert iteration into SFT windows with per-family window shares (`rl/rollout_distillation.py`), template withdrawal at
window boundaries (`rl/source_withdrawal.py`), memetic stages, the BF16 native SFT trainer with its transitions,
capacity proofs, recovery exports, DiLoCo and LR multiplier, CoreWeave cloud/teacher scripts, the dev16 watcher,
`behavior_trend.py` and the controller restart/rewind tools. Kept: the SGLang/native actor (`serve_native_checkpoint.py`,
`pop_rl_actor.py`), the collector, harness and scorer (`rl/task_rollouts.py`, `run_bonsai_queue.py`), `rl/held_eval.py`,
the task pool and `rl/curriculum.select_tasks`, the native server's gradient endpoints (`train/native_graph_loss.py`) and
checkpoint loading (`train/checkpoint_policy.py`). Lessons that still apply:
- A round's collector bundle was pruned after its update and w5 lost 85 successful episodes that way: keep the
  episodes an experiment trains on before pruning anything.
- Muon orthogonalizes `g + beta*m`, so its step has full size whatever the gradient's scale. w5's RL gradient was about
  1/1000 of the momentum buffer the SFT window left, so each RL round took 7-9 full-size steps along stale SFT
  momentum. Check an update's direction against its gradient before blaming the objective.
- On the GB10 a device-to-host copy into freshly allocated pageable memory runs at 0.05-0.1 GB/s (13 GB/s into
  memory the CPU already touched): prefault or pin host buffers before copying checkpoints.
- On the corrected SFT mix a backbone LR of about 2e-5 drove the model into a "repeat the failing action, ignore the
  error" attractor, and `--boundary-weight 32` put 1-9% of the objective on a few marker tokens until the model
  answered `return_result` with a native state. Never upweight a few tokens without per-token evidence.

## Task selection (engineering, 2026-11-03/04)

`rl/curriculum.select_tasks` (selector `natlang.rl-task-curriculum/2`, `exclude_ids` for the held set) starts new
families at their median level and shifts practice by outcome spread. It fills `class_quota` (effect_state 0.34) from
tagged rows: construct shares are enforced at selection, never by dropping pool candidates. It ranks over-visited rows
behind fresh ones (a ranking, never a drop). Measure the sampled mixture with `scripts/rl_tasks/simulate_selection.py`
and task-supply repetition with `scripts/rl_tasks/measure_repetition.py`.

## Task supply and meta-learning (owner, 2026-11-03)

The 2026-11-03 recipe review (`plans/neuralese/RECIPE_REVIEW_20261103/`) found that scripted templates carried about
78% of SFT loss and that RL tasks were 98% one wrapper generator. Step counts and loss on such a mixture are not
capability progress.

Owner (2026-11-03): fill measured behaviour gaps primarily with on-policy RL, not more teacher demonstrations.
Small-model (Haiku) subagents author task programs and exact oracles, prompted for real depth and diversity, and
qualified by reference/negative/alias replay. Authoring is continuous. Rotate domains and problem types. Seed designs
from existing agentic and knowledge-reasoning datasets (never protected or held inputs) and also invent original ones.
Measure repetition against the existing pool before admitting each batch. Never pad scarce behaviours by repetition.
Verified successful rollouts may self-distill into SFT.

Meta-learning is the memetic path (owner, 2026-11-03). The backbone, executing a natlang improvement program, studies
a program's invocations and failures and authors or revises its guidance, skills, prompts and adaptation choices
(including LoRA bank fits, used as a service). It is trained by on-policy RL on "improve program P" episodes, rewarded
by held-world task gains under exact oracles. This is the long-term direction. Its training returns as a controlled
addition once plain updates improve held-out success, and it is promoted by held-family task outcomes.

## Neuralese compaction is the intended runtime design (owner, 2026-11-03)

Owner intent: Neuralese compacts and summarizes large values throughout the
natlang runtime, including parameter/argument listings, large inputs, tool and
service outputs, collections and documents rendered into model context. It also
compacts context between model runs, carrying learned compressed state rather than
full text. Training data includes bgkit-derived (and other) sources transformed
into a Neuralese bottleneck: a Neuralese-prompted summarizer/extractor producer
returns Neuralese to a parent lambda/consumer that must use it for the task outcome.
The compaction and summarization operators are directly learned operators,
trained parameters rather than only prose prompts. Exact KV/prefix reuse is a
separate performance cache, not compaction. Track remaining gaps in the fallow
register until each is live in runtime, data and training.

Directly learned operators (owner, 2026-11-03) start from text: initialize each
operator (compaction, summarization/extraction, guidance, skill and other
prompt-embedding operators) from its natural-language prompt, embedded in the
real rendered call, then let it morph through direct learning on sufficient
dedicated data. Each operator needs its own adequate training data and measured
use, never a nominal parameter with a handful of rows.

Neuralese span basis (owner, 2026-11-03): every Neuralese span, soft state or
learned operator output starts from a good token sequence: input-specific,
target-relevant content the consumer needs, never a constant cue, instruction or
placeholder shared across inputs. Use gold/checked summaries or supporting spans.
Where none exist, use a deterministic mechanical summary (structure outline,
question-selected extractive spans, exact identifiers and values, length scaled to
input), recording seed provenance. Then morph through end-to-end consumer-outcome
learning with annealed text supervision. A span whose content is the same for
every input, or that ablation cannot distinguish from shuffled state, is a defect.

## Built capabilities must not lie fallow (owner, 2026-11-03)

The SGLang actor was built and integrated, then left unused for days behind a
bespoke exactness bar while RL rollouts ran about 100x too slowly on the native engine.
When the slowness surfaced, the intended solution was not raised. Standard
practice (fast-engine sampling, trainer-recomputed logprobs, truncated
importance correction, divergence rejection) was never considered. Rules:
- Keep `plans/neuralese/FALLOW_REGISTER.md` current: every built capability that
  is not in use, with its blocker, impact, next action and owner. Review it between
  experiments; its "Why these went fallow" pattern list is the checklist.
- When any symptom appears (slow, failing, poor learning), first check what we
  already built for it (register, plans, AGENTS.md) before diagnosing from scratch,
  and state that option explicitly.
- A qualification bar may not block an otherwise-ready improvement indefinitely.
  If a blocker has had no active work for 48 hours, either resolve it, replace the
  bar with the standard-practice mechanism that keeps correctness safe, or record a
  deliberate exclusion with its reason. Strictness that prevents use is not safety.
- A capability leaves the register when an experiment has measured it and it is in
  use, or when it is excluded with a reason. A commit alone is not use.

## Platform and archived findings (owner, 2026-11-09)

CoreWeave quota was exhausted on all four accounts on 2026-11-06 and is not expected back. Training runs on the DGX
(learner, SGLang actor, held eval), with Pop as a second actor and evaluator. Engineering findings from 2026-10-15 to
2026-11-02 (checkpoint/custody/capacity/numerics incidents, the Neuralese shared-state architecture, reward-policy
details, generation diversity rules) live in `plans/neuralese/ENGINEERING_FINDINGS.md`. Read the relevant section
before changing that code.

# Main goal: keep moving forward (owner, 2026-10-11)

The main goal in everything is to keep the work moving. Run the next experiment promptly and keep the machines doing
useful work for the pinned experiments (collection, training, held eval). Don't be overly cautious and don't wait for
the owner. If something is sensible, do it, record why and keep going. Raise with the owner only what is truly theirs.

- **A finding is not done until it is acted on.** A defect or a better setup becomes the change of the next
  experiment. If it is a correctness bug in a running experiment, stop that experiment and rerun it. A
  recommendation alone is an unfinished task.
- **Verify, don't assume.** Claims about state, ownership or other agents must come from a check made now:
  processes, coord activity, files, logs. Work whose author shows no activity (no process and no coord message in
  the last 24 h) is orphaned. The supervising agent owns it: review it and integrate or deliberately discard it.
  Don't address it to absent owners.
- **Diagnose surprises immediately.** An unexpected number, failure or regression gets a root-cause investigation
  in the same session, not a note for later.
- **Degradation means bug.** Don't reject an approach because our implementation of it failed (owner, 2026-10-11).
  A worse loss, degraded behaviour or an unexplained numerical gap is a bug until proven otherwise: compare against an
  exact reference, localize the divergence, fix it, and re-measure before judging the approach. When something doesn't
  make sense, run a wide, varied dragnet: data and masking, rendering and templates, dtype and upcasts, caches and
  carried state, kernels, optimizer state and resume, gradient flow, loss weighting, evaluation code, and differences
  from a known-good path. Record what was ruled out.
- **Checks protect results.** Keep checks that protect task quality, numerics, held-data separation, data quality,
  recoverability and runtime function. Base-relative KL/agreement is telemetry, never a stop. Keep launch checks
  nonmutating: preflight once and let the real entry run its own checks. Never run an administrative entry in a run
  directory first (it can leave records that block the real run).
- **Code identity is provenance.** Every run, checkpoint and actor records its git commit and dirty flag
  (`common/code_provenance.py`). Refuse a wrong policy SHA, model/tokenizer identity or artifact schema/version.
  Outside an experiment, services run from the checkout `/home/werg/natlang` (Pop: `git pull`).
- **Comparable curves only.** Lower aggregate loss across changing samples, weights or quantization points is not a
  learning curve. Task outcomes on the pinned held set are.

## Work lands on main, not in worktrees (owner, 2026-10-29)

Shared-main commit hygiene: create new commits scoped to your own paths (`git commit -- PATHS`; a bare `git commit`
also takes whatever other agents staged). Never amend, reset or rebase shared main while another strand can commit;
HEAD may have changed since your last read. Reconcile divergent published history with a normal merge, preserving
both strands, rather than force-pushing.

No work may be lost or abandoned in worktrees: do work directly on main in the canonical checkout, staging only your
own paths. A worktree is a short-lived tool, not a home for work. Merge its commits to main promptly and remove it once
its commits are on main and nothing live or recorded depends on it. The pinned copy an experiment runs holds no
commits and is removed when the experiment ends.

## Agents decide; nothing waits on the owner (owner, 2026-10-29)

Operational and technical decisions such as restarting learners, restarting a
lineage from base, changing the batch or objective, or redoing work are **not**
the owner's decisions. Every agent has the autonomy to do the work the best
possible way and must not block anything on the owner. Decide, record why,
brief the other agents through coordination, and keep going. Agents still
coordinate with each other on shared resources; that is not a reason to wait
for the owner.

## On-policy collection and explanatory prompts (owner, 2026-10-29)

Collect from the actual checkpoint being updated, preserving actor/checkpoint identity and sampled token
probabilities. Historical teacher continuations and a different checkpoint's attempts are not current-policy rollouts.
RL uses the full explanatory prompt and complete tool schemas so correct idiomatic behaviour is accessible: valid
examples of awaited per-item `nl`, exact sorting/filtering/counting in code, `iterateOn` with stopping/effect checks,
direct use of inputs/services, typed concise child answers and `finish:true`. Bind the actual prompt,
tokenizer/template and runtime in both collection and training. Never put protected answers or held trajectories in
that guidance. Compact prompts are a later explicit robustness target. Measure program/template/world repetition in
what is actually trained; missing identifiers are unknown coverage, not zero repetition.

## LFM3 direction: no cross-model dependence, adaptation after a trained backbone (owner, 2026-10-28)

Strike all dependence on another model: no cross-model VAE/shared-dialect stage and no waiting for Mellum. No
additional teacher generation: use the parts of existing data that fit, and exact oracles with on-policy RL where
demonstrations are missing. Once the backbone improves, the goal is everything the tuning system needs (program tunes,
native soft states, LoRA mixtures/bases), the training corpus for the learned optimizer (metalearner), and training the
metalearner itself. Program-tuning executions and metalearner targets are collected on a backbone that has shown held
gains, never on one that has not; the machinery and oracle supply may be prepared in parallel. Order: plain
fine-tuning, RFT and GRPO until held gains, then controlled additions (Neuralese, program adaptation, metalearner).

## Lean codebase and economical testing (owner, 2026-10-28)

Always eliminate legacy infrastructure outside the core direction: code, tests,
docs, recipes and registry entries. Git history is the archive, so keep a lean
codebase with clean abstractions. Preserve source/training data, protected held
data and live recovery dependencies; never keep code "just in case".

Test economically: run the targeted tests for a change, merge, and debug problems
when they appear. Do not rerun suites on a base commit to tell pre-existing
failures from new ones. Fix every failure you see whatever its origin; when a
failing test only covers legacy infrastructure, remove the test and that code.

## Individual-program tuning (owner, 2026-10-26)

The tuning lane (`plans/TUNING_OPERATIONS_HANDOFF_20261026.md`, `plans/PROGRAM_TUNING_RUNS_20261026.md`) measures how far
each program improves through hierarchical adapters, native soft states, SFT and RL. It runs on a backbone that has
shown held gains. Continue saved adapter/soft-state optimizer histories when improving the recipe. Bind round journals
to the actual recipe, student base, initial parent and execution limits. Keep related invocation worlds on one split,
select on development and confirm on untouched confirmation worlds. Report actual base, SFT and RL task outcomes; saved
artifacts and zero-variance rollout groups are not capability lift or policy updates. Distill verified training-world
executions through ordinary source admission, preserving exact actor and split custody.

Checkpoint I/O is part of GPU supervision: track actual commits and independent exports separately from enqueue
acknowledgements, and keep cache offload or fixture preparation from starving a checkpoint write.

## Learned LoRA mixtures and bases (owner, 2026-10-27)

Make direct component-mixture and native soft-state fitting an operational part
of program adaptation. Free hierarchical LoRA fits can supply bank growth and
behavior targets, but they do not establish that mixture training or execution
has occurred. Mixture labels must come from an actual fit in the named bank,
with task outcomes measured through the composed adaptation.

Train the shared LoRA bases and component bank alongside the mixture writer,
coefficient head and native guidance in the shared learning stage. Explicitly
enable `train_basis` and `train_bank` in its operational recipe rather than
inheriting their disabled code defaults. Individual program label fits may pin a
bank snapshot; after a shared bank update, publish its actual new snapshot and
refit/rebind affected coefficient labels. Verify actual basis/component changes
and downstream behavior and preserve optimizer history. Do not describe target
extraction alone as training the learned mixture system.

## Core memetic learning and legacy removal (owner, 2026-10-23)

Memetic learning and soft skills are a core user capability and a training goal. Learn
stored native Neuralese guidance from multiple specific invocations of the same
program, measure actual task outcomes on distinct development and confirmation
invocations, and keep iterating the guidance. Keep related worlds on one side of the
split. Do not claim capability lift from stored artifacts, admission, source checks or
gradient tests alone. Its training enters as a controlled addition (see the top section).

Use the current positional KV and hybrid-state architecture in training and
inference. Stored guidance must condition the real executing model. Retain text and
tokenizer provenance for re-encoding when the executing checkpoint changes, and reuse
compatible state efficiently across repeated and concurrent calls. Guidance authoring
keeps exact actor provenance. The model being trained authors its own memetic guidance;
external teachers supply codebases and verified trajectories, not guidance (owner, 2026-10-31).

The system has no deployed-user backward-compatibility requirement. Remove legacy
adapters, compatibility shims and fallbacks, migrate their current callers and
supported functionality to the native path, and remove obsolete public entrypoints,
docs, tests and discarded-run registry entries. A legacy adapter hiding missing
functionality is work to implement, not a reason to retain the fallback. Preserve
usable source data and protected evaluation data. An explicit unsupported error is
useful during migration but is not completion of missing functionality. Keep actual
live recovery dependencies until their replacement is verified.

## Neuralese training and the LFM RL curriculum (owner, 2026-10-16, 2026-10-31)

Owner intent: actively instill Neuralese use. Mine broad task sources (with outcome oracles) for RL and
self-distillation, and prompt for Neuralese where it helps: tasks and prompt variants declaring `Neuralese<...>`
child results, per-document/per-item views, fan-in to a parent. Don't over-focus on Neuralese to the detriment of
task performance. Neuralese enters as controlled additions: Neuralese-declaring task variants first, then
whole-graph Neuralese gradients and a use bonus for correct episodes only, each measured on its own.

When Neuralese gradients are trained, the update covers the entire actual graph: preserve sampled and forced
producer turns, consumer imports, exact occurrence bindings and downstream outcomes, with gradients through the real
producer/consumer dependencies. Never turn an in-episode producer into a detached external input.

Keep an executable LFM RL task curriculum, starting with tractable verified natlang execution and growing into tool
use, multi-step work and communication. Use actual task outcomes to adapt difficulty; keep protected evaluation separate.

# Machine ownership and coordination

Owner, 2026-11-09: the coord inbox is retired. No Pop agent runs; the DGX supervisor drives Pop directly over ssh
(actor and evals). Do not check `coord.py inbox` in monitoring cycles. The coordination rules below
apply again only if a second concurrent agent is started.

- Use `/home/werg/natlang` as the canonical development checkout on DGX and Pop.
- Synchronize code through small commits pushed often to `origin/main`, using
  ordinary git: pull, merge and rebase as usual (owner, 2026-10-11), and keep the
  checkout current with main. Commit your own work promptly and stage only your
  paths; don't discard or stash away changes you didn't make.
  `scripts/git_publish.py` remains available as a helper. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.
  Reconcile coverage against every entry of the exact upstream inventory, with
  explicit member/index bindings; sequential directory labels are not a coverage
  proof. Keep conversion, admission, staging and actual learner exposure separate.
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

# Storage hygiene and discarded work (standing owner order, 2026-10-11)

Owner clarification (2026-10-16): delete pretty much everything except what is
necessary to carry on. This includes remote W&B transfer/export storage. Keep
the latest independently verified recovery state, explicitly active diagnostic
or handoff parents, useful source/training data and protected held data. Prune
superseded checkpoint exports continuously after replacement durability is
verified; do not accumulate every periodic full checkpoint. Write the exact
deletion list first and verify removal. Never delete an entire mixed data and
checkpoint run merely because most of its bytes are obsolete checkpoints.

Amendment (owner, 2026-10-13): make deletions less eager. With the external disk
available for space, lean towards moving data to `/mnt/external` rather than
deleting it when unsure.

- Delete discarded approaches and runs: their checkpoints, optimizer states,
  exports, frozen runtime copies, diagnostic outputs, obsolete recipes and run
  tracking. Do not archive them, move them to another disk or machine to avoid
  deleting them, or retain them because an old note or manifest references them.
  Remove obsolete registry entries, manifests and handover tracking as well.
- Keep the active lineage and its required parents, reusable training/source
  data, and protected/held evaluation data. Distinguish these from discarded run
  artifacts before deleting; check live process references and current recipe
  dependencies. A regression still needs diagnosis before rejecting an approach.
- Deduplicate exact copies and prune superseded intermediate conversions,
  rendered packets, review previews, replays and caches. Keep one current useful
  representation with its source provenance, rather than every attempted version.
  Use existing deduplication and retention tools; preserve content hashes and
  logical paths where a current consumer needs them. Never hardlink mutable files.
- Prune rejected outputs in general. Retain one only for a concrete current use:
  an assigned repair, an unresolved bug investigation, or an actual failure/DPO
  training input. Record that use and its consumer; "might be useful someday"
  is insufficient. After the use ends, remove it. Keep independently useful
  correct rows from a partly rejected batch; remove bad rows and redundant copies.
- Check storage during the normal monitoring cadence and after large runs or
  conversions. Clean autonomously under this order. Brief the other machine so
  discarded artifacts and obsolete registry entries are removed there by its
  owner too, rather than silently surviving in mirrors. Record the current
  decision and reclaimed totals, without building another archive of discarded
  approaches. This order supersedes older blanket preservation/PRUNE-ASK rules
  for explicitly discarded work.
- These deletions are expressly desired and authorized by the owner (2026-10-11):
  deleting discarded run directories, checkpoints, exports, caches and their
  registry/manifest entries on either machine is wanted. Do it without asking for
  confirmation; list the exact paths in a script first, then delete.

# Natlang-native work (owner rules)

These rules apply to every agent on every machine for the product: the natlang
language, runtime, compiler, tools and apps. Paste the short form into any subagent
brief. Reasons and history are in the linked documents. The training and RL stack is
the exception: it uses standard tooling (an HF training loop, the SGLang server) and
minimal crisp code.

**The main rule: think in natlang-native terms.** Before building anything in the
product (an app, a runtime component, a compiler pass, a tool, a policy, a review step),
first ask how it would be a natlang program: typed natural-language functions,
callable folders, services for the outside world, and crisp code only where
exactness or the outside world requires it. Start from that design and justify
every piece of crisp code. Do not start from conventional code and add natlang
afterwards. The rules below are consequences of this one.

- **Policy is natural language; mechanism is crisp.** Any product component that decides
  something (which tier serves, what to fuse, what to promote, which task runs,
  how to recover) is a natlang program. This includes natlang's own runtime and
  compiler, not only applications. Crisp code is for the outside world
  (services), the CLI, exact durability (atomic commit), exact verifiers of what
  the stages produce, and plumbing. When a policy is hot, give it one interface
  with a crisp and a natural-language implementation selected by a setting
  (`ts-host/src/runtime/pluggable.ts`); crisp may be the default.
- **Port end to end, unit by unit.** A system "in natlang" has its whole logic as
  natural-language stages decomposed like the real system. For every part,
  record whether it becomes its own function, one instruction, implicit (rare)
  or a crisp helper, and why. Spell algorithms out as numbered steps over named
  data for small models. Write DECOMPOSITION.md first, then implement and merge;
  owner review happens afterwards (queue: `plans/OWNER_REVIEW.md`).
- **Decide model-facing changes on first principles** (owner, 2026-11-12). Bug fixes and
  structural changes to what the model sees (lost or erased actions, history that differs from
  what was sampled, needless prompt rewrites, limits below the model's capacity) are fixed
  directly, never gated on an A/B. Record each in `plans/MODEL_FACING_CHANGES.md`; held-out
  evaluation of the next checkpoint shows any regression.
- **Teach in errors, not prompts.** Never spell out a wrong form in a prompt; it
  primes the model to produce it. Handle wrong forms in the runtime and teach the
  fix in a one-sentence error. Prefer making standard JS idioms work over new
  names; be very circumspect with new model-facing names. Every user-facing
  semantic change updates the covering reference of `skills/natlang-authoring` or
  `skills/natlang-integration` in the same push, with recovery advice for each new
  error; native-training and training-application notes go to
  `docs/NATIVE_RUNTIME_NOTES.md` and `docs/SEMANTIC_APPLICATION_NOTES.md`, not the
  skills (`skills/README.md`).
- **No premature limits.** Termination is structural (no while loops; recursion
  on a smaller argument; `iterateOn` with a measure or a natural-language
  predicate), not budgets or default timeouts. Add a resource limit when a
  problem shows up.
- **Apps:** decide on a snapshot (model calls), then apply a pure bounded commit.
  Effects are data, timers are state, and derived values form a DAG. Stay close
  to JS semantics.
- **One implementation per concept.** Before adding a mechanism, look for the
  existing one (specializer, call store, pluggable, refinement shadow, shared
  hashing in `natlang_neuralese/common`) and extend it.
- **Supervise the model's own outputs** (owner, 2026-10-28; retires the
  2026-10-07 whole-trajectory supervision rule). Assistant actions and typed
  answers are the targets; prompts, instructions, inputs, tool/service output
  and harness feedback are context with zero loss weight. Breadth comes from
  more natlang domains, not from anchors or broad mixing; anchors never cap
  learning.

# Working practice on shared machines

- On DGX every build, test and probe runs through the memory ledger
  (`scripts/memory_ledger.py run --unit <family>-<id> --budget-gb N -- ...`; units are asynchronous, so wait and read
  the log). Priority: `--priority high` for training/GPU runs, `low` for data generation and check-main, default
  `medium` for smokes/evals. Never report tests as passed when the ledgered run did not happen.
- `memory_ledger.py adopt` changes accounting only; it does not resize a unit's `MemoryMax`. Restart a stopped
  transient service with a fresh unit name.
- Environments: tests and data use `/home/werg/natlang/.venv-neuralese/bin/python` with
  `PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang`; Node builds use the Node24 bin under
  `/home/werg/.local/share/natlang-dev/`; `os.pidfd_open` guards need host `/usr/bin/python3`. Ledger units do not
  inherit the caller's shell environment.
- Experiment launches use the experiment's pinned copy as the ledger `--workdir`, with `NATLANG_REPO`/`PYTHONPATH`
  bound to it; other launches use `/home/werg/natlang`.
- Do not add or remove hardlinks to source/manifest/index files while live readers stream them (ctime changes
  invalidate strict reader checks).
- Shared checkout: stage only your own paths; never push mid-rebase/merge; never stage `node_modules` symlinks.
  When resolving a merge conflict keep both sides unless one was removed on purpose.
- Before pushing run `scripts/check-main.sh --quick`; after runtime/test changes run the full `check-main.sh` under
  the ledger. Main stays green.
- Standalone tests get a disposable `NATLANG_HASH_CACHE` and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
  (`check-main.sh` sets both).
- Stop processes by systemd unit name or explicit PID, never by pattern. Every agent may stop, kill, restart,
  relaunch or remove any process, unit, ledger job or container without asking (owner, 2026-10-12); record why.
  Containers get `--name <ledger unit>` and are declared with `memory_ledger.py run --container <name>`.

# Neuralese training foundation

LFM3-1.7B is the backbone (owner, 2026-10-30; replaces the LFM2.5 QAD base and Neuralese direct v1 rules). Its port,
data curation, quantization codecs and evaluation packets are in `plans/neuralese/LFM3_RECIPE_20261028.md`. How it is
trained now is the top section of this file and `plans/neuralese/APPROACH_REVIEW_20261111.md`. A resume restores its
run's exact checkpoint and optimizer/RNG state; never substitute base weights into a resume. Qualification is actual
task behaviour, numerics and recoverable state, not similarity to the base.
