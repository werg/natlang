# LFM3 scale-out: local first, then DiLoCo SFT on 4 sandboxes, then multi-machine RL (owner, 2026-10-28)

## Execution plan after the owner restructure (2026-10-28, later)

Committed first-run decisions (DiLoCo outer optimizer, 4 learners directly, async RL pattern, phase-1 scope) are in `LFM3_COMMITTED_DECISIONS_20261028.md`, which supersedes the hedges below.

Parallel preparation now (independent agents/ledger units): port A (backbone) completion; part C fork merge and
converter cutover; Pop corpus recovery (Pop back online, ssh alias pop-os); LFM3 conversion of all usable data and
the prioritized joint-SFT mixture (natlang execution + Neuralese in one stage); sandbox runtime (prebuilt x86_64
sm_120 fork binaries, bootstrap without compilation, DiLoCo/eval/actor templates, boot validation); score-centered +
JOLT+ RL learner (default); default-on audit; oracle repairs; tuning lane + learned optimizer on LFM3 with
teacher-free program supply.

Throughput: time-boxed optimization passes (first box: one working day of wall time from the moment the LFM3
backbone lands on main) on DGX and on one real sandbox GPU, since the DGX Spark (GB10, unified memory, aarch64) does
not represent the RTX PRO 6000 sandboxes. Training starts at the end of the first box with the best measured
configuration; further optimization continues during training and is adopted at safe update boundaries through
recorded transitions (optimizer/RNG/exposure preserved).

Order: joint SFT (DiLoCo, 4 sandboxes) -> JOLT+ RL -> tuning lane on the trained checkpoint -> metalearner (bank-init,
corpus, outer training with train_bank/train_basis, meta-RL). No cross-model stage.


Owner 2026-10-28: "first iron out the kinks on this machine, then transition to DiLoCo SFT with 4 sandboxes, then
figure out a good pattern for swift RL, using several machines in concert where useful". This authorizes four
one-GPU learner sandboxes (plus the CPU DiLoCo coordinator) for the LFM3 SFT stage once the local phase passes.
CoreWeave credit remains opaque (AGENTS.md): bounded lifetimes, measured throughput, usage monitoring, private
full-state exports before long runs.

## Phase 1: local on DGX (now)

Exit criteria, each measured on the real path:
1. Port qualification (`LFM3_PORT_20261028.md`): HF-reference logits/gradients, one-pass graph vs cached execution,
   native serving vs llama.cpp on the same GGUF, native held-v3 reference.
2. Output-only objective (workstream E) declared in the LFM3 recipe; per-category loss telemetry visible.
3. LFM3 data twins for the first stage's sources; idiom audit decides the allocation.
4. A short local SFT run from LFM3 on the recipe's first stage: step time and memory at the real batch geometry,
   actual parameter updates (incl. FP32 sparse marker rows), save/restore/continue preserving optimizer/RNG/exposure,
   reply/action CE falling separately from reasoning CE, development-packet outcomes measured through native
   serving before and after. Only after this does anything go to the cloud.

## Phase 2: DiLoCo SFT on 4 sandboxes

Reuse `train/diloco` + `scripts/coreweave/diloco.py` (`DILOCO_COREWEAVE_20261027.md`; never yet run on CoreWeave).
LFM3-specific sizing to settle with the phase-1 numbers:
- Sync volume: 1.7B trainable FP32 = ~6.8 GB per full sync (~3.4 GB BF16 wire). Choose `local_steps` so a round's
  transfer and merge are a small fraction of its compute, using the existing fragment plan (streaming fragments
  overlap compute). If an adapter-first stage trains only adapters/new rows, sync is small; full-backbone stages
  need the measured choice.
- Learner memory: full FT with AdamW FP32 masters ~27 GB + activations on a 96 GB GPU; set the RAM minimum from the
  measured host footprint (not the 2 Gi default shape).
- Outer optimizer: short stages showed Nesterov overshoot on few rounds; start with averaging or low momentum and
  compare held/dev against the local run at equal exposure.
- Bring-up: 2-learner smoke for one or two rounds (identity, merges, export to a continuation checkpoint, restore on
  DGX), then 4 learners. Exports: coordinator root and learner-0 full state privately exported at a cadence; DGX
  keeps the verified recovery copy.
- Selection: DGX evaluates each exported global version on the development packet through native serving while the
  learners continue (DGX is the evaluator/selector, not a learner).

### Sandbox launch kit (prepared 2026-10-28, validated on one GPU sandbox)

Nothing compiles in a sandbox any more. Prebuilt, content-addressed artifacts (registry
`scripts/coreweave/prebuilt-artifacts.json`, objects under `/home/werg/data/coreweave-prebuilt/cas`, mechanism
`scripts/coreweave/prebuilt.py`), all for image `nvcr.io/nvidia/pytorch@sha256:f71fd67f…` (26.09 linux/amd64; the
launcher refuses another image for image-bound artifacts):

| Name | sha256 | Bytes | What |
|---|---|---|---|
| `llama-cpp-neuralese-sm120` | `cb3da79d…48d1f2c` | 198,727,937 | fork `neuralese-lfm3` 85a2e7167, CUDA `120a-real`, static ggml/llama: llama-server, state-span-dump, logits-dump, bench, quantize, cli, perplexity. Built in 178 s on 33 CPUs |
| `llama-cpp-neuralese-fat` | `84bb29d8…89c9840` | 472,412,318 | same for `90a-real;100a-real;120a-real` (334 s); only for non-Blackwell assignments |
| `wheelhouse` | `eb8025dc…bebda2b` | 50,306,185 | the 31 exact wheels requirements.txt adds to the image (bound to its blob); installs offline in 12 s |
| `ts-host` | `bcf3c272…0b07ab2` | 31,927,069 | dist + production node_modules, built on DGX (JavaScript only), bound to its source trees |

(Full digests in the registry.) Build: `scripts/coreweave/jobs/prebuilt-build.json` (CPU sandbox, 32 CPU/64 Gi, lived
~13 min). Use: `bundle.py plan ... --prebuilt llama-cpp-neuralese-sm120 --prebuilt wheelhouse --prebuilt ts-host`.
The bootstrap unpacks them, exports `NATLANG_LLAMA_SERVER`/PATH, skips pip and npm, and installs the zstd CLI the NGC
image lacks. A changed requirements.txt or ts-host source refuses the stale artifact at plan time with the rebuild
command.

Templates `scripts/coreweave/jobs/lfm3/` (fill with `render.py VALUES.json OUT`): `diloco-sft.run.json` (4 learners,
quorum 3, plain averaging `--outer-lr 1.0 --outer-momentum 0.0` to start, `local_steps` 128 provisional),
`diloco-coordinator.job.json` (CPU 8 / 48 Gi, FP32 globals), `diloco-learner.job.json` (one GPU + 64 Gi provisional
RAM minimum, 300 Gi volume, hourly snapshot schedule), `dev-eval.job.json` (dev packet v2, 128 cases),
`actor.job.json` (HTTPS service), `serve-check.job.json`. Placeholders are only the recipe/stage, parent checkpoint,
trainer arguments, bundles and ids. A `recovery` block makes `cwctl.launch` start `recovery_snapshots.py` as a ledger
service; `cwctl stop` ends it. Snapshots stay supplemental (all three Mellum r6 captures were FAILED); learner 0 and
coordinator private full-state exports remain the recovery route.

Validation `lfm3-serve-check-20261028` (sandbox 691221d4, 2026-10-28): create 08:55:13 -> running 23 s (with a 100 Gi
volume) -> inputs pushed from DGX 08:56:46 (281 MB, ~66 s, the dominant cost) -> fetch+verify+prebuilt install
08:57:12 -> llama-server healthy on the official BF16 GGUF 26 s after the job started (24 s of it the 3.4 GB HF
download) -> correct `eval` tool call (prompt 588 tok/s, decode 255 tok/s). Create-to-ready about 2.5 min, versus
~45 min of compilation in `lfm3-1p7b-heldv3-20261027`; the sandbox lived 3 min 20 s and was stopped after
collection. Assigned: RTX PRO 6000 Blackwell (sm_120, 97,887 MiB), **2 CPUs** (Xeon 8562Y+), 32 Gi cgroup limit
for a `{"gpu": 1, "memory": "32Gi"}` request, as in the previous eval. torch 2.14 / CUDA 13.4 matmul probe at
4096^3: BF16 350 TFLOPS; FP8 tensorwise 2.02x, rowwise 1.93x; FP8 128x128 block scaling unsupported (SM90 only);
NVFP4 1x16 block + tensorwise (`F.scaled_mm`) 2.82x, block-only `_scaled_mm` 1.79x.

Open before the real launch: the LFM3 recipe/stage and phase-1 parent (placeholders); the learner RAM minimum and
`local_steps` from phase-1 step time and host peak; whether 2 assigned CPUs starve learner data loading (AGENTS.md
keeps CPU unspecified, so measure it in the 2-learner smoke and raise it if it binds);
`tests/test_neuralese_diloco.py::test_two_http_learners_train_and_merge` currently fails (final loss 1.9-2.1 vs the
0.21 bound, 2 of 2 runs on 21518f2a1) and needs diagnosis before the DiLoCo smoke.

## Phase 3: multi-machine RL pattern (to settle after phase 2 measurements)

Default candidate: **DiLoCo-RL**. Each sandbox runs the existing native whole-graph RL round loop (its own actor
serving the current local policy, collector, verifier, GRPO learner) on its own stable task shard, and the DiLoCo
coordinator merges pseudo-gradients every H updates. Rollouts stay on-policy within a sandbox with no weight
broadcast over WAN, and it reuses both existing mechanisms. Needs: the RL learner calling `DilocoLearner.after_step`,
curriculum shards that keep outcome variance per shard, and a global curriculum view.

Alternative if learner-side compute dominates: asynchronous actor/learner (K actor sandboxes serving published
policy versions with captured behaviour log-probs and bounded staleness, one learner consuming groups continuously).
Costs a weight broadcast per version (small only for adapters) and importance correction.

Score centering (TRAINING_INSIGHTS_20261028.md section 5) changes the trade-off: it stays stable under quantized
samplers and, with truncated IS, under staleness of tens of steps. That makes the asynchronous actor/learner pattern
cheaper than assumed (actors need not wait for every version, and can serve the deployed NVFP4/FP8 policy), and it
also protects DiLoCo-RL against the drift between each sandbox's serving engine and its trainer.

Choice by measurement: rollout seconds vs update seconds per group, fraction of zero-variance groups, sync cost,
and measured trainer/sampler drift.
