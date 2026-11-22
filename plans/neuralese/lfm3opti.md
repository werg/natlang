# LFM3-1.7B training at every scale we can reach: Colab G4 as a third machine, one LoRA learner core

**Status:** v3, the supervising agent's revision of the v2 brief (v2 is commit 0f3a9c684; its audit findings and
contracts are kept where they hold). Every item below is decided and required; there are no optional or deferred
items. **Date:** 2026-11-22. **Code base:** `werg/natlang` main after bfc5a7a7d.

**Goal.** More held-v3 gain per day, measured as expert-iteration cycles per day times gain per cycle, across every
machine we have: DGX GB10, Pop (RTX 4060, offline today), and Colab G4 (RTX PRO 6000 Blackwell, 96 GB, as many
sessions as the CU budget runs). Today one cycle takes ~14-17 h: collection 6 h, teacher, fill and build ~1 h, SFT
4.5-7.3 h on the GB10, and a held eval of ~3 h on Pop. The target is a cycle in **3-4 h**, with two lineages in flight
at once (one on the G4s, one on the DGX).

**What changed from v2, and why.**

1. **The whole cycle, not only the learner.** The SFT is ~40% of a cycle; collection and held eval are the rest. Decode
   is memory-bandwidth bound (`ACTOR_THROUGHPUT_20261112.md`). The G4 has ~1.6-1.8 TB/s against the GB10's 273 GB/s, so
   a G4 SGLang actor collects and evaluates several times faster, and that is the largest single lever on wall time.
2. **LoRA is the parameterization (owner 2026-11-22).** It is `lora+rows` for SFT and RL, with no comparison eval.
   The full-parameter path is removed from `exp0_sft.py` and `grpo.py` when the core lands (git history keeps it).
   Until then, phase 0 trains with today's code on the G4.
3. **Defects are fixed now** (§3), not at milestones.
4. **No gates.** Each piece ships in small commits with targeted tests and the AGENTS.md opt-out, enters running work
   at the next cycle, and the next held eval judges the bundle. v2's correctness checks are tests that ship with the
   code.
5. **v2's non-goals were decided on first principles.**
   - **Required:** Blackwell kernels, FP8 frozen-base GEMMs, and no recomputation (§6).
   - **Dropped:** DiLoCo multi-worker training, SGLang adapter slots, learner/actor overlap and L4 profiles, each with
     its reason (§6).
6. **Integrated everywhere.**
   - One remote-machine mechanism for Pop and Colab.
   - One learner core for SFT, GRPO, the probes and the soft-operator fits.
   - One merged export for all four engines.
   - §7 is the inventory.

---

## 1. Machines and placement

**Placement and the training loop are specified in `plans/neuralese/async-train.md`**, which supersedes this section,
§2 items 4-5, §4 and §6 "Learner/actor overlap" where they differ. In brief:
- the G4 runs the LoRA learner (back-to-back SFT rounds) and an actor beside it under MPS;
- the DGX collects, runs the teacher and fill, builds, merges, publishes and runs the held eval (BF16 series);
- collection is continuous from the newest policy.

The rest of this section is the original v3 placement.

| Machine | Why | Work |
|---|---|---|
| **Colab G4** (96 GB GDDR7, Blackwell) | ~6x the GB10's memory bandwidth and several times its tensor throughput; preemptible | the main lineage's collection actor, SFT and held eval (G4 engine series, §4). Extra sessions take collection and held-eval shards. |
| **DGX GB10** (128 GB unified) | always on; holds all data, scorer DBs and the archive | the episode collectors and scorers for every actor; teacher, fill and build; the second lineage (its own collect -> SFT on the GB10); Neuralese, operator and diagnostic work; any stage whose Colab host is gone |
| **Pop** (RTX 4060 8 GB) | cheap, slow | forward-only probes and Pop fill |

**L4 is excluded.** Its ~300 GB/s bandwidth equals the GB10's, so it adds nothing the DGX doesn't already do.

**Placement.** `expert_iteration.py` gets an `ExecutionProfile` per stage (`dgx`, `colab_g4`, `pop`). Each profile
names:
- the host;
- budgets (VRAM and host RAM separately on discrete GPUs; the ledger's combined budget on the GB10);
- engine flags and data paths.

A stage runs on its profile's host. When that host is unreachable for 30 minutes, the stage runs on the DGX. This
generalizes the Pop pattern built in b14f425de, and Colab preemption is handled the same way.

## 2. Phase 0 (now): Colab as a machine, today's code

1. **Reach Colab from the DGX.**
   - `scripts/colab/bootstrap.py`, a notebook cell, installs Tailscale (userspace networking) and sshd, clones the
     pinned commit, creates the venv, builds ts-host, fetches the base model and reports ready. The DGX supervisor then
     drives the session over ssh, as it drives Pop.
   - `pop_rl_actor.py` becomes the remote-actor module with per-host config. Pop and Colab share one implementation of
     "serve a checkpoint on a remote GPU, attest it, stop it".
2. **Data and artifacts.**
   - Resumable rsync over Tailscale: ~1.1 GB of chains and a 3.5 GB checkpoint per cycle each way.
   - Every checkpoint, run state and result goes back to the DGX as soon as it exists. Colab's disk is scratch.
3. **SFT on the G4.** `exp0_sft.py` as it is today, with `--activation-gb 80` so nothing is recomputed. The SFT state
   is saved to the DGX every 15 minutes, so a preempted session costs at most 15 minutes.
4. **Actor and collection.**
   - The G4 runs the SGLang actor at the engine's full running cap. The episode collectors stay on the DGX, beside the
     scorer DBs and task data, and reach the actor over the tunnel, as Pop fill does.
   - Per-call network latency (tens of ms) is small against decoding hundreds of tokens per call.
5. **Held eval on the G4** at 4 samples per task (§4).

## 3. Defects fixed now

| Defect | Where | Fix |
|---|---|---|
| **Split-brain serving after an in-place rebind.** Requests routed to the native engine (adapters, text writes) keep running the start checkpoint while SGLang serves the new policy. | `serve/sglang_actor.py:236-252` (`rebind_in_place`), `_routes_native` | Reload the native engine from the same export at every rebind. `/v1/neuralese/info` reports each route's policy sha, and GRPO checks every route after each sync. |
| SFT checkpoints record the source step, not the learner step | `exp0_sft.write_checkpoint` | Record `learner_step` and the source separately. |
| AdamW state is allocated for every parameter, frozen or not | `exp0_sft.load_model`, optimizer factory | Allocate only for trainable leaves. |
| One full-FT memory constant (`ACTIVATION_BYTES_PER_TOKEN_LAYER = 88e3`) | `lfm3_fast_train.py:772` | Replace it with a table keyed by (layer kind, device), filled from the measured `cuda_peak_by_phase_gb` of the LoRA core. |
| `*_gb` labels mean GiB in some places and GB in others | telemetry | Report GiB everywhere and rename the labels. |

## 4. Held evaluation (superseded: the held series runs on the DGX, BF16 weights, 4 samples; async-train.md §1)

- **One series.** The G4 series becomes the held series. It starts by measuring the current reference policy once,
  and every later candidate pairs against it. Pop's series closes with its last result.
- **4 samples per task.** At 2 samples, differences under ~8 points are noise and most verdicts come out neutral.
- **The adaptive interval (a993c0f87) stays the schedule.** The owner wants fewer evals, and on the G4 each one costs
  far less.
- **A held set larger than one session's share** is sharded across the G4 sessions that are up.

## 5. The learner core: LoRA with trainable Neuralese rows

v2 §3 and §5 remain the specification, with the changes below.

- **One core:** `train/lfm3_learner.py`, `train/lfm3_fast.py` (moved from `scripts/neuralese_data/lfm3_fast_train.py`)
  and `train/learner_state.py`. It is shared by:
  - `exp0_sft.py` and `grpo.py`;
  - the probes `behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py` and `passthrough_check.py`;
  - the `operator_soft_states` fits;
  - the learner bench.
- **Parameterization: `lora+rows`.**
  - Frozen BF16 base, with FP32 factors and FP32 AdamW state.
  - The embedding/head rows of the Neuralese marker and override tokens are trainable (FP32 masters for those rows
    only).
  - Head, other embeddings and norms are frozen.
- **Factored projection:** y = W0·x + s·B(A·x).
  - A and B are FP32 leaves stored unscaled; the scale s is applied once.
  - The low-rank branch runs in BF16 on tensor cores with FP32 accumulation. Strict FP32 at r=128 would cost most of
    the saving.
  - No dense merge during training.
  - Bundles are leaf-only and passed flat through `checkpoint(use_reentrant=False)`, because the prefix tree runs
    several backward passes.
  - Gradients accumulate directly into the FP32 leaves.
- **Learning semantics preserved exactly** (v2 §5):
  - weighted global denominators;
  - the nucleus and full-vocabulary laws;
  - Ambient SFT (`p_clean` 0.75, T 2.5);
  - GRPO `natlang_grpo_v1`, with distinct current, behaviour and reference policies; the reference is a frozen factor
    snapshot;
  - packing isolation in attention and conv state;
  - prefix reuse within one step;
  - differentiable Neuralese pass-through.
- **Tests shipped with the core:** an eager FP32 oracle and the v2 §10.1 table (adapter bypass, frozen drift, prefix
  and packing parity, Neuralese gradient flow, reference aliasing, merge scaling, route attestation, resume replay).

| Setting | Value | Why |
|---|---|---|
| SFT rank | r = 128 on all 9 projection families, alpha = r | A cycle supervises ~16M loss tokens. *LoRA Without Regret* finds LoRA matches full FT while capacity exceeds the data; rank 32 is RL-sized. |
| RL rank | r = 32 | Policy gradient carries a few bits per episode; ranks of 1-32 match full FT there. |
| LR | 10x the full-FT value: SFT 5e-5, RL 1e-4 | The paper's 10x rule. |
| Lineage | Each cycle merges its factors into the base in FP32, rounds to BF16 once (the next cycle's start policy) and restarts with B = 0 | Capacity then has to hold one cycle's data, not the lineage's. |

## 6. Speed work, decided on first principles

**Required:**

1. **No recomputation and larger packs.**
   - LoRA frees the ~27 GB of full-FT state, so every activation stays resident on the G4.
   - On the GB10, the freed memory goes to `--activation-gb`: sft-v8's backward was 3.1x forward because of
     recomputation.
   - Packs grow to fill the memory, and the effective batch stays at 32k weighted target units.
2. **Blackwell kernels.**
   - LFM3 attention runs through FlashAttention/cuDNN SDPA built for sm_120.
   - The causal-conv kernel gets a launch configuration for the G4's SM count.
   - `torch.compile` runs with token-length buckets, and its recompile and graph-break counts are logged.
   - The compiled paths were tuned on the GB10; on a different SM architecture they leave throughput unused.
3. **FP8 frozen-base GEMMs** (torchao float8 on Blackwell tensor cores, G4 and GB10 alike).
   - The base forward and input-gradient GEMMs run in FP8, while adapters, trainable rows, loss and optimizer stay in
     FP32/BF16.
   - Without weight gradients, base GEMMs are nearly all of the learner's FLOPs, and FP8 doubles their tensor rate.
   - The FP32-master rule protects trainable weights; the base is frozen.
   - Shipped test: logprob agreement with the BF16 path on the probe set within the BF16-vs-FP32 tolerance.
   - Under LoRA, the BF16 merged export is what the actors serve, so train-serve parity is checked against it.
4. **Publication for GRPO.** GRPO publishes a merged BF16 export (base plus the step's factors, merged in FP32) with
   the publish, attest and rollback machine of v2 §7.1. The native-route reload from §3 is part of it, and so is
   attestation of all four routes.
5. **Sharded collection and held eval.** Every G4 session that is up runs an actor. The collectors split the cycle's
   tasks across them by group, and held episodes by task. This is how extra sessions buy wall time, because collection
   and evaluation split cleanly by task.

**Dropped:**

- **DiLoCo multi-worker training.** With LoRA, FP8 and no recomputation, a cycle's SFT on one G4 takes about an hour,
  while collection is the larger share and parallelizes by task (item 5). Averaging workers over the internet would
  add sync machinery to shorten the shorter stage.
- **SGLang adapter slots.** A merged 3.4 GB export loads from `/dev/shm` in seconds, against GRPO steps of minutes.
  Slots would also need a new serving feature on all four engines.
- ~~Learner/actor overlap on the G4.~~ Reversed in async-train.md: with continuous collection from the newest policy
  there is no stage dependency, and the G4's decode bandwidth is its strongest asset. The learner and an actor share
  the G4 under MPS.
- **L4 profiles.** The bandwidth is the GB10's (§1).

**Cost telemetry.** Each cycle logs CU spent, wall time per stage and canonical chain tokens per CU, so the budget is
spent where the time goes.

## 7. Integration inventory: every consumer of what changes

| Mechanism | Components | Notes |
|---|---|---|
| Remote GPU host (serve, attest, measure, fall back) | `pop_rl_actor.py` becomes the remote-actor module; `expert_iteration.py` (popbase, held, Pop fill, collect, SFT placement); `grpo.py --held-remote`; `ops/grpo-v4/pop-held.sh`; `stop_pop_fills` | One implementation for Pop and Colab. |
| Learner core | `exp0_sft.py`, `grpo.py`, `behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`, `operator_soft_states`, learner bench (`--replay`) | Probes load the merged export or the core's effective policy, never the bare base. |
| Parameterization and export | `tune/targets.py` (the one target inventory), `tune/export.py`, `export/native_hf.py`, GGUF for managed llama-server and in-page wllama, the native engine, SGLang | Every engine serves the merged BF16 export, so engine parity is unchanged. Held eval and the `lfm-latest` profiles serve it. |
| Execution profile and budgets | `expert_iteration.py`, `memory_ledger.py` (DGX), a VRAM/host admission shim on Colab, `ops/ei/launch.sh` | |
| Held series | `rl/held_eval.py` (`series()`, `compare`), `expert_iteration.reuse_held`, `latest_base`, `ops/current-policy.json` | The series key includes the engine host. |
| Resume and provenance | `common/code_provenance.py`, `train/learner_state.py`, the verdicts' `since_held` and `reference` | Colab runs record SKU, usable GiB and driver versions. |
| Docs | AGENTS.md "Machines" and "Current approach" (Colab G4 becomes a machine; LoRA is the method's parameterization), `docs/ENGINE_PARITY.md`, `FALLOW_REGISTER.md`, DECISIONS | |

## 8. Work order (four concurrent strands; none blocks training)

- **Strand 1, Colab:** bootstrap, remote-actor module, `ExecutionProfile` with DGX fallback, SFT and actor on the G4,
  the G4 held series, and sharded collection and held eval. The next cycle that can use it runs its SFT on the G4.
- **Strand 2, defects (§3):** native-route reload with route attestation, learner-step provenance, trainable-only
  optimizer state, the memory table and GiB labels.
- **Strand 3, learner core (§5):** extract the core; add adapter bundles through all four fast paths and the MLPs;
  ship the oracle tests; switch SFT and GRPO to `lora+rows`; remove the full-parameter path.
- **Strand 4, speed (§6 items 1-4):** resident activations and packs, Blackwell kernels and buckets, FP8 base GEMMs,
  and GRPO merged-export publication.

Each strand commits small steps with targeted tests to main. Each change enters running work at the next cycle.
Results, with the measured numbers, go to DECISIONS.md.
