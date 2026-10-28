# LFM3: committed decisions for the first run (owner, 2026-10-28)

Owner: "we want to first go with a YOLO run with our best foot forward, instead of budgeting time for lots of
experiments. Of course, smaller experiments that are really very, very bounded are fine if they really are very
important." Each hedge in the LFM3 plans now has one decision. The alternative is recorded with the signal that
should trigger switching to it. Switches happen live at a safe update boundary, preserving optimizer, RNG and
exposure; they are not a restart. Where an older plan doc still says "measure", "compare", "pilot" or "candidate",
this file supersedes it.

| # | Was hedged | Decision now | Fallback, and when |
|---|---|---|---|
| 1 | Stage 1 trains Neuralese rows + adapters first, widening to the backbone later | Full backbone from update 0: MuonSR on all matrices, AdamWSR on norms, conv kernels and learned rows (port A routing) | Adapter-first, if held loss or dev outcomes collapse in the first ~500 updates and no bug is found |
| 2 | Learning rate "tuned once at local geometry" | Backbone matrices peak 2e-5 (AdamW scale; Muon update scaled 0.2·sqrt(max(m,n))); learned rows/adapter 1e-3; weight decay 0.1; warmup 150 updates; then constant, with EMA 0.999 (eval reads the EMA). DiLoCo batch transfer by the Muon rules | Halve the lr on a loss spike or growing grad norm that is not a bug; double it if dev and loss are flat after 2k updates while rows move |
| 3 | Reasoning share "start 10%, measure" | 10% of assistant loss, fixed | 5% if reasoning grows long or rambles; 20% if reply quality drops where reasoning is skipped |
| 4 | Data allocation "starting, adjusted from audit"; tiering | The owner's priority tiers govern order and weight only (deep/diverse/natlang/Neuralese/reducer first). No caps of any kind (near-duplicate, family and category caps removed 2026-10-28 after they dropped 87% of converted data); one epoch over all usable data. No rebalancing runs | Reweight live if dev shows a specific idiom lagging |
| 5 | Idiom gaps "must come from new teacher generation" | No teacher generation (owner). Gaps come from oracle-built training-split tasks (`decide` tasks, curriculum) in SFT and from JOLT+ RL | None needed |
| 6 | Phase 1 local shakedown, then a dev comparison before the cloud | Phase 1 = the throughput time box plus a ~200-update local run that checks real updates, save/restore/continue and falling reply CE. No before/after dev comparison run: the base eval exists, and DiLoCo dev evals start with round 1 | None |
| 7 | DiLoCo 2-learner smoke, then 4 | Launch 4 learners directly (quorum 3); verify after round 1 that the merge, export and DGX restore of the global version work, then let it run | Drop to 2 learners only if round 1 shows a mechanism fault |
| 8 | DiLoCo outer optimizer: averaging vs Nesterov, "compare against local run" | Plain averaging (outer lr 1.0, momentum 0) for rounds 1-10, then Nesterov 0.9 with outer lr 0.7 (DiLoCo defaults). `local_steps` is computed from measured step time so transfer + merge ≤ 5% of a round (128 if the rule allows). No comparison run | Back to averaging if global dev or loss regresses after the switch |
| 9 | Learner data loading on 2 assigned CPUs | Pre-tokenize and pack shards on DGX, so learners do no tokenization; CPU stays unspecified (AGENTS.md) | Raise the CPU request with the owner only if the loader-stall telemetry shows starvation |
| 10 | Phase 3 RL pattern "to settle after measurements" (DiLoCo-RL vs async) | Asynchronous actor/learner: 3 actor sandboxes serving the NVFP4 policy from the llama.cpp fork (top-k capture), 1 learner sandbox (JOLT+ twin forward, privileged teacher). Score centering + TIS absorbs staleness and quantization | DiLoCo-RL if the learner is the bottleneck or fork top-k capture slips |
| 11 | JOLT+ "pilot against score-centered GRPO at equal tokens" | JOLT+ is the RL loss from round 1, with beta_S 1.0 and beta_T 0.5, k = 128 | Score-centered GRPO alone if the KL terms dominate or diverge; beta_T 0.25 if the teacher tilt destabilizes |
| 12 | Per-head Muon for tuning fits "measured there" | Not used; normal per-matrix Muon everywhere | none |
| 13 | Metalearner mixture: A/B of the old grid+head vs component-token distribution | Component-token distribution (null token, gain per target group, masked softmax, top-p execution) is the only parametrization; old grid/head path deleted (git history keeps it) | Restore the old path from git only if fits under the new form cannot reach the old fits' quality and no bug is found |
| 14 | Harness ergonomics: about 48 samples per change | One combined variant (all candidate fixes) against the baseline, about 48 samples each on the dev packet; drop a single fix only if the combined variant regresses | Bisect the fixes only on a regression |
| 15 | Each stage "qualifies" at BF16/FP8/NVFP4 on dev | Monitoring, not a gate: dev evals at all three precisions every N DiLoCo rounds on DGX | none |
| 16 | Mellum line (legacy sweep asked whether to retire it) | Retire it: remove Mellum-specific trainer/operator/recipe code last in the sweep. Keep its final exported checkpoint (step 39257) and all data | Revive from git history |
| 17 | RL actor engine (llama.cpp fork only) | Owner 2026-10-28: maintain a vLLM or SGLang fork with the full Neuralese feature set (KV/conv-state import/export, markers, privileged blocks, exact top-k capture, `decide` with imported state, adapters/soft states). Benchmark result (`LFM3_RL_ACTOR_ENGINE_20261028.md`): fork **vLLM 0.32.0**, >10x the llama.cpp fork on RL-shaped load. Port order: LFM3 model file + compact top-k (Neuralese-free episodes), then Neuralese spans/conv state/producer adapter/markers/privileged blocks, then decide, LoRA mixtures, guidance rewind. Actors move to it; the llama.cpp fork stays for local/browser serving | Keep llama.cpp actors if the benchmark shows no benefit |
| 18 | Generic coding data (union r11 external-expansion-codeagent, swe-rebench, first-wave without natlang structure; ~300M supervised tokens) | Owner 2026-10-28: a small coding mix, about 10% of joint-SFT supervised tokens, a diverse verified sample in its own lowest tier, interleaved, declared as `coding_mix` in the mixture policy. Caps removed: tiers are ordering and weights only | Raise or lower the share live if dev shows coding helping or crowding out natlang |

Bounded checks still on (each is minutes, and each protects the run):
- the step-0 identity/parity checks;
- the first-round DiLoCo restore check;
- the real-update and changed-coordinate telemetry;
- the parity tests for new graph paths (`decide`, privileged teacher).
