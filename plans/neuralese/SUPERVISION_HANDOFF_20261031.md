# LFM3 supervision handoff — 2026-10-31 (Claude → Codex)

Written by the DGX supervising Claude session (coord identity `dgx-claude-5c7cf3fb`) at ~08:30 CET.
Live numbers go stale fast: **re-verify every claim below against processes, files and logs before acting**
(AGENTS.md "Verify, don't assume").

Authoritative logs (read these first, newest at the bottom):
- `/home/werg/data/lfm3-rl-dgx-20261029/ops/decisions.log` — every RL decision, cutover, incident (146+ lines).
- `/home/werg/data/dgxops/claude-supervision-actions-20261029.md` — hourly supervision entries.
- `/home/werg/data/dgxops/claude-deletions-20261029.txt` — every deletion/move I made (with reasons).

## 1. Where things stand

### Decisions in force (owner-approved or made under owner's full authority, 2026-10-31)
- **SFT step608 line dropped.** SFT-aligned paired eval (program-header rendering, base vs step608): base 32 vs
  step608 17 of 51 paired, gained 6 / lost 21, McNemar p=0.006 → regressed. Result:
  `/mnt/external/natlang-development-data/archive/data/lfm3-eval/decision-dgx-20261030/sft-aligned-partial51.json`.
  RL starts from base (`fresh-init`). Future SFT only on the repaired broad mixture (v4) with the serving prompt.
- **RL from base on DGX, two lineages:** `jolt-r1` (JOLT+ GRPO) and `sd-r1` (self-distillation/RFT).
  RL-vs-base after ~6 rounds: 4 vs 5 of 9 paired (no signal; lineages barely moved).
- **Neuralese bonus (owner amendment in AGENTS.md "Whole-graph RL"):** group advantages centered on task outcome
  only; +0.1 added afterwards to correct episodes whose graph actually passes Neuralese state producer→consumer
  (338c5c2ba). Failures never get it; correct non-Neuralese episodes never go negative. SD weights such rows 1.1×.
- **Broad RL task source v2** `runs/natlang-rl-broad-external-20261031-v2/cases.jsonl` (32,947 cases, 23 external
  sources + inline-curriculum slice; held dev/confirmation packets in the protection union). **v1 is withdrawn**
  (45 held-SFT-related + 55 unloadable DataBench cases); never train on it. sd-r1's v1 round-8 attempt was
  abandoned untrained (files kept in `sd-r1/round-000008`, `ABANDONED.json`).
- **Learner `--max-graph-forward-tokens` stays 180000** (120000 deferred 39% of episodes). `checkpoint_seconds`
  raised 300 → 1800 in both lineage configs (15 GB saves every 5 min + external offload filled NVMe).
- **Cloud continuation lineages (`jolt-r1c`/`sd-r1c`) not launched** (no RL lift); cloud starts 4 fresh lineages.

### DGX RL (dir `/home/werg/data/lfm3-rl-dgx-20261029`)
- Runner: ledger unit named in `ops/runner-unit` (currently `lfm3-rl-par-9`, 100 GB, high), running
  `scripts/alt_par.sh 6` (both lineages; updates serialized by `update_lock.sh`; per-lineage logs
  `logs/par-jolt.log`, `logs/par-sd.log`). Budget file `ops/runner-budget-gb` = 100.
- State at handoff: jolt-r1 round 6 learner update running (~64 of ~77 episodes; ~3 h per update);
  sd-r1 round 8 (v2, `round-000008-attempt-1`) collected, its update waits.
- **Pending automatic action**: `/home/werg/.claude/jobs/8018b402/tmp/restart-after-jolt6.sh` (PID was 3351943)
  waits for jolt-r1 `trained_rounds >= 7`, stops the runner, runs the cloud helper's GPU measurement
  `ops/gpu-window-command.sh` once as ledger unit `rl-gpu-window-1` (≤25 min; output `logs/gpu-window-1.log`,
  result `debug/gpu-window-1/result.json`), then relaunches as `lfm3-rl-par-10`. **If this Claude session ends,
  that script dies with it** — do it by hand: after jolt r6 trains, stop the runner unit, optionally run the
  window script, relaunch:
  `python3 scripts/memory_ledger.py run --unit lfm3-rl-par-N --budget-gb 100 --priority high --wait 3600 --detach -- /usr/bin/bash -c "exec $W/scripts/alt_par.sh 6 > $W/logs/par-N.log 2>&1"` and update `ops/runner-unit`.
- Runtimes (immutable git-archive copies, `scripts/build_runtime.sh COMMIT /abs/path/FROM_RUNTIME`):
  `runtime-5b9a0a6c5` (jolt r6, bound), `runtime-afb13bb47` (sd fallback), `runtime-cd5416516` (current main fixes:
  put_bytes resume, typed-text-result deferral). Both lineages have `pending-switch.json` → `runtime-cd5416516`
  + v2 cases, 16 rows/round, 32768 context. Switching: `scripts/cutover.sh RUNTIME 'CONFIG_JSON'` (preflight per
  lineage; a bound round that fails preflight finishes on its old runtime).
- Bugs fixed this session (all on main): put_bytes removed by 23627b32f broke resume (afb13bb47); uncaptured
  typed-text-result Neuralese writes failed whole rounds → learner defers those groups and preflight skips them
  (c1c929610, cd5416516). **Open engineering gap:** differentiable text→state lowering so those episodes train.
- Memory incidents: global OOM 04:52 and ledger-floor stop 05:17 when jolt's update (~63 GB CUDA + 14 GB RSS) overlapped
  sd collection (~30 GB) plus other jobs. Keep both lineages in ONE runner unit; don't split them into separate
  units (claims don't fit). `measure_runner_peak.sh` now refuses to write an empty budget.
- Disk incident 07:30 (NVMe 6 GB): checkpoint_offload copies of deleted checkpoints held space on the saturated
  external disk. My watch script killed such copies; checkpoint interval raised. Keep NVMe > 40 GB.

### Episode-length/throughput findings (owner asked)
- Learner packs causal prefixes (round 6: 31.7M unshared → 7.2M packed tokens, 4.4×); system prompt + tools ≈
  6–7k tokens/episode (~7%): **the full prompt is not the cost.**
- Cost = long trajectories on the old curriculum: jolt r6 30% success, median 23.5k sampled tokens, failures 33k,
  35/83 budget exhaustion, 59/83 hit per-turn length. Broad v2 (sd r8): 74% success, median 4.6k tokens (5× cheaper);
  clutrr/babi ramble to length limits; spider/bird 100% (no GRPO signal → down-weight/raise difficulty for JOLT+).
- Learner ~5.4k tok/s GPU-bound. Levers: cross-episode (group) prompt prefix sharing (~6%), prep speed (11 min
  single-core before GPU), long-graph memory (cloud helper measuring checkpoint-input file offload), cloud GPUs.
- Analysis script: `/home/werg/data/lfm3-rl-dgx-20261029/scripts/ep_length.py ROLLOUTS...`; restart script copy: `scripts/restart-after-jolt6.sh`.

### Cloud (CoreWeave)
- Accounts `cw1..cw4` (`~/.config/natlang/coreweave-accounts/cwN.env`; never print or pass keys into sandboxes).
  Default account FROZEN. Cross-account test passed. **Every GPU create is refused `CWSANDBOX_GPU_NOT_ALLOWED`
  ("organization is not entitled to create GPU sandboxes") — owner/CoreWeave must enable GPU entitlement.**
  Zero sandboxes running. Spend is tracked in GPU-hours only (prices unknown): `cwctl.py accounts`.
- Owner policy: no caps, no early retirement, balance spend; on the first explicit quota error evacuate all
  accounts' state to DGX; normal export cadence.
- Tooling: `scripts/coreweave/rl_cloud.py`, `rl_cloud_sandbox.py`, `rl_cloud_autopilot.py`; plan
  `plans/neuralese/LFM3_RL_CLOUD_20261031.md`; run file `/home/werg/data/lfm3-rl-cloud/run.json` (v2 cases).
  Units `rl-cloud-autopilot-5` (one GPU probe/hour, then smoke + jolt-c1/sd-c1/jolt-c2/sd-c2, urgent coord on
  entitlement) and `rl-cloud-export-4` (exports + blob retention via retention_leases). Inputs are content-addressed
  W&B blobs (cases `54ad7d9e`, assets `e442206c`). SGLang fork rebased on Liquid's LFM3 PR (`8f2fd5894`, pin
  `8be682c94`); **GPU parity checks still owed** once a GPU exists.

## 2. Work in flight (helpers were Claude subagents of this session — they stop when it ends)
Pick these up from their files/units, not from the agents:
1. **Natlang codebase RL tasks (owner priority):** tasks that hand the model sizable *natlang* codebases — `nl`
   lambdas, `.nl` functions with callable folders, directory reducers (`kind: directory-reducer`,
   `folder.apply`), `natlang.d/`, iterateOn folder reducers — to run eval/read_code and per-file/per-item
   `Neuralese<...>` child calls with fan-in; plus authoring/repairing natlang code judged by executed tests/oracles.
   Sources: `codebases/`, `applications/`, skill assets, tuning/memetic oracle programs, generated variants.
   Target 25–35% of RL rows; new source version, related worlds on one split. Builder work under
   `scripts/rl_tasks/` (e.g. `ts_analyze.mjs`) and `/home/werg/data/broad-rl-tasks-20261031/`.
2. **Broad RL v3:** builders committed (c6c18fa5c: TabFact, WikiSQL, MSR-SQA, ConvFinQA, MultiHiertt, ToolACE,
   NESTFUL, tau-bench, APIGen-MT; 3ae1bf1fa multi-answer puzzle verifiers; 66a9b1374 >2^53 numbers → re-admit
   55 DataBench cases). Verification units `broadrl-v3-*`. Plan/notes `plans/neuralese/BROAD_RL_TASKS_20261031.md`.
   Not done: decontaminated big math sets; KodCode/verifiable-coding need a Python runner; measure live-model
   false rejections of correct paraphrases from RL rejects.
3. **Successor SFT mixture v4:** `plans/neuralese/LFM3_DATA_COVERAGE_AUDIT_20261031.md` (build commands at the end),
   policy `training/neuralese/mixtures/lfm3-joint-sft-mixture-policy-v4.json`. Running low-priority: render queue
   `lfm3-data-queue-loop5`, coding v2 (`lfm3-cov-coding-v2`, 12/repo), port records (`/tmp/cov-port-s4.sh` →
   conversion of all 45 families), acquired sessions (`lfm3-cov-acq2-0/1` → `acquired-sessions-lfm3-20261031-v2`).
   Defaults: `--serving-entry mix`, family loss shares 40/40/12/8 with cap 8. Open: D12 regrade hold; D24 exports
   only on a CoreWeave `/work` volume; communication data far below its 40% share. native-sft-v13-r2 source was
   recovered to `/mnt/external/natlang-lfm3-data-20261028/native-sft-v13-r2-recovered-20261031/`.
4. **Broad Neuralese graphs:** `plans/neuralese/NEURALESE_BROAD_GRAPHS_20261031.md`; units `bgraph-*`
   (rest27 relaunched with 24 GB after 8 GB OOMs; big2 converter fixes; resumable phases 49b061636, 6a2e1e9a2).
5. **Learner efficiency (cloud helper):** profile/cut the ~11-min single-core prep in `rl/native_stage.py`;
   train long graphs at lower peak without deferral (checkpoint-input file offload; exact gradient parity).
   GPU window result → `debug/gpu-window-1/result.json`.

## 3. Owner-pending items
- GPU entitlement for cw1..cw4 (blocks all cloud work).
- Delete `/mnt/external/outputs/proto1` (24 GB, Feb checkpoints)? `sudo rm -rf
  /mnt/external/natlang-native-directory-20261002/model /mnt/external/natlang-vllm-qwen36` (root-owned).
- AGENTS.md still has stale Mellum/Qwen/VAE sections. Memetic authoring: self vs JOLT+ teacher.

## 4. Gotchas learned this session
- Ledger: `run` without `--wait` fails fast when headroom is short; with `--wait --detach` it queues as
  `<unit>-waiter` and preempts lower resumable units. Requeued stopped units can be re-admitted later —
  `memory_ledger.py cancel --unit X` when retiring one. Low jobs are guard-stopped first under pressure.
- Large copies to `/mnt/external` must use the bounded copy (`archive_to_external.py` / `checkpoint_offload`
  verified_copy); an unbounded rsync built dirty cache and contributed to the OOM. External disk is I/O-saturated.
- `archive_to_external.py` leaves symlinks; mark live dirs with `.keep-on-nvme` (done for both RL dirs). A `find`
  for deletion must not traverse into the archive (I deleted the archived base GGUF that way; regenerated via
  `claude-regen-base-gguf-1` to `/mnt/external/natlang-development-data/archive/data/lfm3-eval/LFM3-1.7B-BF16.gguf`
  — verify it completed).
- Never edit a running bash script in place (restart it). `build_runtime.sh` second arg is a runtime *path*.
- py-spy needs sudo here; use timing logs/cProfile.
