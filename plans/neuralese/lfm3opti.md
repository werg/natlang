# LFM3-1.7B training at every scale we can reach: Colab G4 as a third machine, a learner core that runs anywhere

**Status:** v3, the supervising agent's revision of the v2 brief (v2 is commit 0f3a9c684; its audit findings and
contracts are kept where they hold). **Date:** 2026-11-22. **Code base:** `werg/natlang` main after bfc5a7a7d.

**Goal.** More held-v3 gain per day. The unit is expert-iteration cycles per day times gain per cycle, across every
machine we have: DGX GB10, Pop (RTX 4060, offline today), and Colab G4 (RTX PRO 6000 Blackwell, 96 GB) plus whatever
else the CU budget buys (L4, more G4 sessions). Today one cycle takes ~14-17 h: collection 6 h, teacher, fill and build
~1 h, SFT 4.5-7.3 h on the GB10, and a held eval of ~3 h on Pop. The target is a cycle in **3-4 h** and **several cycles
in flight at once**, so expert iteration turns over 4-6 times a day instead of about once.

**What changed from v2, and why.**

1. **Scope: the whole cycle, not only the learner.** The SFT is ~40% of a cycle; collection and held eval are the rest.
   Decode is memory-bandwidth bound (`ACTOR_THROUGHPUT_20261112.md`). The G4 has ~1.6-1.8 TB/s against the GB10's
   273 GB/s, so a G4 SGLang actor should collect and evaluate several times faster, and that is the largest single
   lever on wall time. v2 left rollouts as a "revisit trigger". Here they are the first deliverable.
2. **No recipe gate in front of the speed.** Our current full-parameter recipe (FP32 masters and moments, ~27 GB of
   state) fits a 96 GB G4 with room for every activation, so it runs there **unchanged on day one**. LoRA is built in
   parallel and judged against it. It earns its place where it is structurally better (RL, preemptible workers,
   multi-worker averaging, cheap references and publication), not as a precondition for using the G4.
3. **Fixes ship now.** The audit found real defects: the native-route split-brain after an in-place rebind, and the
   learner-step provenance. Under AGENTS.md ("A finding is not done until it is acted on") they go to main this week,
   with tests and opt-outs, independent of any Colab or LoRA milestone.
4. **No exit gates, smokes or engineer-week estimates holding work back.** Each piece ships with targeted tests, turns
   on by default and is read through its telemetry; the next held eval judges it ("No pre-launch smoke gates a
   feature"; "Training never waits"). v2's correctness checks stay as **tests that ship with the code**, not as phase
   gates.
5. **The non-goals list becomes a lever list.** FP8 base GEMMs on Blackwell, Blackwell attention kernels, multi-worker
   training and SGLang adapter slots are in scope. They are ordered by expected gain and built once the core runs on G4.
6. **Integrated everywhere.** One remote-machine mechanism serves Pop and Colab alike. One learner core serves SFT,
   GRPO, probes and the soft-operator fits. One export reaches all four engines. §7 is the inventory.

---

## 1. Machines and placement

| Machine | Strength | Default work |
|---|---|---|
| **Colab G4** (96 GB GDDR7, Blackwell, discrete) | ~6x the GB10's memory bandwidth; several times its BF16 throughput (measure); preemptible, session-limited | the cycle's critical path: collection actor, SFT, held eval (new engine series, §4) |
| **DGX GB10** (128 GB unified) | always on, large memory, all the data on NVMe | the second lineage or arm, teacher and fill, build, GRPO polish, Neuralese and operator work, every diagnostic; it takes over the critical path whenever Colab is away |
| **Pop** (RTX 4060 8 GB) | cheap, always on when reachable | forward-only probes, a second held engine for A/A checks, Pop fill |
| **More Colab** (L4s, more G4 sessions) | parallel capacity paid in CU | more arms, collection shards, multi-worker SFT (§6.3) |

**Placement is a driver decision, not a code fork.** `expert_iteration.py` gets an `ExecutionProfile` (`dgx`,
`colab_g4`, `colab_l4`, `pop`) per stage. Each profile names the host, budgets (VRAM and host RAM separately on
discrete GPUs; the ledger's combined budget on the GB10), engine flags and data paths. A stage runs where its profile
says; when that host is unreachable it waits a bounded time and then runs on the fallback. This is the pattern just
built for Pop held evals (b14f425de), generalized to every stage. Colab preemption is the normal case, not a failure.

## 2. Phase 0 (start now): Colab as a machine, existing code unchanged

None of this waits for LoRA or the learner refactor.

1. **Reach Colab from the DGX.**
   - Tailscale inside the runtime (userspace networking) and sshd, so the DGX supervisor drives it exactly as it drives
     Pop today.
   - A bootstrap notebook cell, checked into `scripts/colab/bootstrap.py`: install Tailscale, clone the pinned commit,
     create the venv, build ts-host, fetch the base model, then report ready.
   - Generalize `pop_rl_actor.py` into a remote-actor module with per-host config, covering Pop and Colab, so there is
     one implementation of "serve a checkpoint on a remote GPU and attest it".
2. **Data and artifacts.**
   - A cycle moves ~1.1 GB of chains and a 3.5 GB checkpoint each way. Over Tailscale (rsync, resumable) that is
     minutes; measure it on the first transfer.
   - Every checkpoint, run state and result is written back to the DGX (the archive) as soon as it exists. Colab's disk
     is scratch. Drive/GCS is a mirror only if the tunnel proves slow.
3. **Run the existing full-parameter SFT on G4.**
   - `exp0_sft.py` unchanged, with `--activation-gb` sized to fit (no recomputation at 96 GB), from the same chains and
     start checkpoint as the DGX would use.
   - Record forward tok/s against the GB10's 2.8-3.0k. This is the first and largest learner win, and it needs zero
     recipe change.
4. **SGLang actor on G4.**
   - Collection runs against it. The episode collectors stay on the DGX and reach it over the tunnel, as with Pop fill.
     Move them to Colab if CPU or latency binds (measure `collector` vs `engine` time).
   - Held eval runs there too, at 4 samples per task instead of 2, once its series is set (§4).
5. **Preemption.**
   - Each stage is resumable today: collection by groups, `held_eval measure` by episodes, and SFT by `--resume`.
     Phase 0 adds an SFT state save every N minutes to the DGX, so a lost session costs minutes.
   - The driver re-places a stage whose host vanished.

**Expected result of phase 0.** Cycle wall time drops from ~15 h to ~4-5 h, and the DGX becomes free for a second
cycle lineage or arm.

## 3. Fixes that ship this week regardless of Colab

| Defect | Where | Fix (default on, opt-out, targeted test) |
|---|---|---|
| **Split-brain serving after an in-place rebind.** Requests routed to the native engine (adapters, text writes) keep running the **start** checkpoint while SGLang serves the new policy. | `serve/sglang_actor.py:236-252` (`rebind_in_place`: "the native engine ... keep[s] the start checkpoint"), `_routes_native` | Reload the native engine's weights from the same export at rebind. Until that lands, refuse native routes after a rebind with a one-sentence error, so no request is silently served by a stale policy. Multi-route attestation: `/v1/neuralese/info` reports each route's policy sha. GRPO checks all of them after every sync. |
| SFT checkpoints record the source step, not the learner step | `exp0_sft.write_checkpoint` | Record `learner_step` and the source separately. |
| AdamW state is allocated for every parameter, `requires_grad` or not | `exp0_sft.load_model` and the optimizer factory | Allocate only for trainable leaves. This is a prerequisite for LoRA and frees memory today wherever anything is frozen. |
| The memory planner has one full-FT constant (`ACTIVATION_BYTES_PER_TOKEN_LAYER = 88e3`) | `lfm3_fast_train.py:772` | Keep a calibration table keyed by (parameterization, layer kind, device), filled from the measured `cuda_peak_by_phase_gb`. |
| `*_gb` labels mean GiB in some places and GB in others | telemetry | Report GiB everywhere new; rename the labels as they are touched. |

## 4. Held evaluation on G4: a faster, better-sampled series

- **A new engine series.** Held results pair only within one attested engine. The G4 series starts by measuring the
  current reference policy once; every later candidate pairs against it.
- **Pop stays a second engine.** It is used for A/A checks, and its series remains valid for its own history.
- **Fast evals mean more samples and less waiting.**
  - Use 4 samples per task as the default (one-sample differences under ~8 points are noise; at 2 samples we often
    call neutral).
  - The adaptive interval (a993c0f87) still applies. Evals become cheap enough that the driver may measure every cycle
    when it is not on the critical path, because it runs on its own G4 session or beside training.
- **Never compare across series.** A pin with a different harness re-measures its reference on G4, as today.

## 5. The learner core (from v2, kept and sharpened)

v2 §3 and §5 are sound, and they remain the specification. In brief:

- **One adapter-aware LFM3 core** (`train/lfm3_learner.py`, `train/lfm3_fast.py` moved from
  `scripts/neuralese_data/lfm3_fast_train.py`, and `train/learner_state.py`), shared by `exp0_sft.py`, `grpo.py`, the
  probes (`behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`) and the
  operator soft-state fits (`operator_soft_states`, which today run their own path).
- **Parameterization is a run setting:**
  - `full`: today's recipe, kept first-class;
  - `lora`: frozen BF16 base, FP32 factors and FP32 AdamW;
  - `lora+rows`: also trains the Neuralese marker and override rows, and optionally norms.
- **Factored projection contract** (v2 §3.3, unchanged):
  - the FP32 low-rank branch runs outside autocast;
  - unscaled leaves, with the scale applied once;
  - no dense merge during training;
  - leaf-only bundles passed flat through `checkpoint(use_reentrant=False)`, because the prefix tree runs several
    backward passes;
  - gradients accumulate directly into the FP32 leaves.
- **Learning semantics preserved exactly** (v2 §5): weighted global denominators, the nucleus and full-vocabulary laws,
  Ambient SFT (`p_clean` 0.75, T 2.5), GRPO `natlang_grpo_v1` with distinct current, behavior and reference policies,
  packing isolation in attention **and** conv state, prefix reuse within one step, and differentiable Neuralese
  pass-through.
- **Tests that ship with the core:** an eager FP32 oracle, and the v2 §10.1 table (adapter bypass, frozen drift,
  prefix and packing parity, Neuralese gradient flow, reference aliasing, merge scaling, route attestation, resume
  replay).

**The recipe, chosen by what the data says rather than by caution:**

| Setting | v3 choice | Why |
|---|---|---|
| SFT rank | **r = 128** on all 9 families, alpha = r (scale 1) | An SFT cycle supervises ~16M loss tokens. *LoRA Without Regret* finds LoRA matches full FT only while capacity exceeds the dataset; rank 32 is RL-sized. Rank 128 costs ~4x r=32's factors, which is still small next to 96 GB. |
| RL rank | r = 32 | RL carries a few bits per episode; *LoRA Without Regret* shows rank 1-32 matching full FT for policy gradient. |
| LoRA LR | **10x the full-FT LR**: SFT 5e-5 (full 5e-6), RL 1e-4 (full 1e-5) | The 10x rule from the same paper. v2's 1e-4 for SFT was 20x. |
| Head, embeddings, norms | Frozen; marker and override rows trainable in `lora+rows` | Neuralese markers must keep learning. |
| FP32 vs BF16 adapter GEMMs | FP32 first. BF16 adapters plus an FP8 base on G4 come as a lever (§6.1). | Correctness first, then speed, with telemetry. |

**How LoRA earns its place.**
- From the first cycle the core runs on G4, the same collection and mix train **two arms**: full on one machine and
  LoRA on the other. Held eval on G4 picks the arm the next cycle starts from.
- This is the bundled engineering comparison the extra compute makes free, not an attribution study.
- If LoRA matches full, it becomes the default where its structure pays: RL, multi-worker training and preemptible
  sessions. Full stays wherever it wins.

## 6. Speed levers on Blackwell, in order of expected gain (build after phase 0, each default-on once it lands)

1. **No recomputation, larger packs.** At 96 GB the full-FT SFT keeps every activation, and LoRA frees ~27 GB more for
   longer packs and larger prefix trees. The effective batch stays the same (32k weighted target units per step).
2. **Blackwell kernels.**
   - FlashAttention or cuDNN SDPA for LFM3's attention layers on sm_120. The compiled attention path was tuned on the
     GB10; check what it lowers to on G4.
   - Check the causal-conv kernel's occupancy.
   - Audit `torch.compile` recompiles and graph breaks, and use token-length buckets (v2 §6.3).
3. **FP8 frozen-base GEMMs under LoRA** (torchao or Transformer Engine, on Blackwell FP8 tensor cores). The base
   forward and the input gradient run in FP8, while adapters, loss and optimizer stay FP32/BF16.
   - This roughly doubles the throughput of the base GEMMs, which are almost all the learner's FLOPs once weight
     gradients are gone.
   - Telemetry: train-vs-actor logprob mismatch on the probe set. Turn it off at a resume if that rises.
   - This replaces v2's "no FP8" non-goal. The FP32-master rule concerns trainable weights; a frozen base is not one.
4. **Multi-worker training (DGX + G4, or several G4/L4 sessions).** Use DiLoCo-style outer averaging: each worker runs
   H inner steps on its shard, then the workers average.
   - With LoRA the pseudo-gradient is the factors (r=128: a few hundred MB), so syncing over Tailscale is cheap. With
     full FT it is 3.4 GB per sync at long H.
   - A worker that drops out is skipped for that round (elastic), and the outer optimizer state lives on the DGX.
   - Check `plans/neuralese/CONTINUOUS_BANK_METALEARNING_20261103.md` and the earlier DiLoCo wiring before writing new
     code. Extend, don't duplicate.
5. **Publication for RL.**
   - Merged BF16 export plus the atomic publish, attest and rollback machine (v2 §7.1), with the native-route fix from
     §3 as part of it.
   - Then SGLang adapter slots (v2 §7.2, packed ranks 3r for fused qkv and 2r for fused w1/w3, derived automatically)
     so a GRPO step's weight sync becomes a small factor file instead of a full reload.
6. **Concurrent learner and actor on G4.** Unlike the GB10, a discrete card with this much memory may overlap
   collection and training usefully. Measure the overlap and default it on if it pays. Don't carry the GB10's
   `--learner-phase-slots 0` across blindly.

**Economics.**
- Log CU/h, cold-start-to-done CU, and canonical chain tokens per CU for each profile (v2 §6.4's formula).
- Spend the CU. A budget that expires unused is the worst outcome. The driver keeps as many Colab sessions busy as the
  budget allows, and the formula decides G4 vs L4 per stage: collection and held eval are bandwidth-bound, the SFT is
  compute-bound, and the cheaper card may win one and lose the other.

## 7. Integration inventory: every consumer of what changes

| Mechanism | Components | Notes |
|---|---|---|
| Remote GPU host (serve, attest, measure, fall back) | `pop_rl_actor.py` becomes a generic remote actor; `expert_iteration.py` (popbase, held, Pop fill, collect, SFT placement); `grpo.py --held-remote`; `ops/grpo-v4/pop-held.sh`; `pop_rl_actor.stop_pop_fills` | One implementation for Pop and Colab, with per-host config. |
| Learner core | `exp0_sft.py`, `grpo.py`, `behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`, `operator_soft_states` fits, learner bench (`--replay`) | Probes load the effective policy or its merged export, never the bare base. |
| Parameterization and export | `tune/targets.py` (one target inventory), `tune/export.py`, `export/native_hf.py`, GGUF export for managed llama-server and in-page wllama, native engine, SGLang | LoRA candidates are merged-exported for every engine, and held eval and user-facing profiles (`lfm-latest`) serve the merged weights. Per AGENTS.md "One serving feature, every engine", adapter-slot serving lands on every engine or gets an `ENGINE_PARITY.md` row naming its blocker. |
| Execution profile and budgets | `expert_iteration.py`, `memory_ledger.py` (DGX), a VRAM/host admission shim on Colab, `ops/ei/launch.sh` | Discrete-GPU budgets are separate from unified-memory ones. |
| Held series | `rl/held_eval.py` (`series()`, `compare`), `expert_iteration.reuse_held`, `latest_base`, `ops/current-policy.json` | Each series records its engine host. |
| Resume and provenance | `common/code_provenance.py`, `train/learner_state.py`, the verdicts' `since_held` and `reference` | Colab runs record SKU, usable GiB and driver versions. |
| Docs | AGENTS.md "Machines" and "Current approach", `docs/ENGINE_PARITY.md`, `FALLOW_REGISTER.md`, DECISIONS | Colab becomes a first-class machine in AGENTS.md. |

## 8. Work order (parallel strands; nothing here blocks training)

- **Strand 1 (phase 0, highest gain):** Colab bootstrap and reachability, the generic remote actor, the SFT and actor
  on G4, the held series on G4, and `ExecutionProfile` with fallback placement. sft-v9 or sft-v10 runs its SFT on G4.
- **Strand 2 (fixes, §3):** the native-route split-brain, learner-step provenance, trainable-only optimizer state, and
  the memory calibration table.
- **Strand 3 (learner core, §5):** extract the core, add adapter bundles through all four fast paths and the MLPs, add
  the FP32 oracle tests, then SFT LoRA, then the two-arm cycle.
- **Strand 4 (levers, §6):** kernels and recompile audit on G4, FP8 base GEMMs, DiLoCo multi-worker, RL publication
  and adapter slots, and the concurrency measurement.

The strands run concurrently; each ships in small commits with targeted tests, and each enters running work at the next
cycle or resume. Results, including negative ones, go to DECISIONS.md with the measured numbers.

## 9. What would change the plan

- **The G4 SFT is not ≥3x the GB10's tok/s:** profile the input pipeline, kernels and recompiles first. If it is still
  slow, put Colab's CU into collection and held eval, where bandwidth is certain to pay.
- **The LoRA arm loses to full at r=128:** check exposure, LR and masks first, then try `lora+rows` and r=256. Keep full
  for SFT and LoRA for RL.
- **Collection is bound by collectors, not the engine:** move the collectors next to the actor (on Colab) or add
  collector processes.
- **Tunnel throughput binds:** stage the mixes and checkpoints through Drive/GCS, and keep the DGX as the archive.
